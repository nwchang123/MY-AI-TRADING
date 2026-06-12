from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

from trading_agent.data.http_utils import fetch_with_retry
from datetime import date, datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from trading_agent.data.moomoo_market import (
    US_OPTION_LOT_SIZE,
    occ_to_moomoo_code,
    parse_us_option_code,
)
from trading_agent.domain.risk import QuoteSnapshot

# Moomoo MY does not entitle US option market data (it costs money), but paper
# AND live option EXECUTION work without it -- only quotes/chains are blocked.
# So option data is sourced from a free, delayed feed and execution stays on
# Moomoo. These providers expose the option-side surface of MoomooMarket
# (is_optionable / option_expirations / option_chain / is_listed_option /
# option_quote) so an instance is a drop-in for ``market`` in the trading cycle.

_USER_AGENT = "moomoo-small-cap-options-agent/0.1 (+option-data)"
_CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
# Tradier sandbox: free developer token, ~15 min delayed, brokerage-grade REST.
TRADIER_SANDBOX_URL = "https://sandbox.tradier.com/v1"


class OptionDataError(RuntimeError):
    """Raised when an option-data source is unreachable or returns bad data."""


@runtime_checkable
class OptionDataProvider(Protocol):
    """Option-data surface the scanner, orchestrator, and validator depend on."""

    def is_optionable(self, underlying: str) -> bool: ...

    def option_expirations(self, underlying: str) -> list[date]: ...

    def option_chain(
        self, underlying: str, start: date, end: date, option_type: str = "ALL"
    ) -> list[dict[str, Any]]: ...

    def is_listed_option(self, option_code: str) -> bool: ...

    def option_quote(
        self,
        *,
        option_code: str,
        expiry: date,
        lot_size: int = US_OPTION_LOT_SIZE,
        now: datetime | None = None,
    ) -> QuoteSnapshot: ...


def underlying_symbol(underlying: str) -> str:
    """Bare ticker for a data lookup: 'US.AAPL' or 'aapl' -> 'AAPL'."""

    return underlying.strip().upper().removeprefix("US.")


def _parse_cboe_timestamp(value: Any, fallback: datetime) -> datetime:
    """CBOE's top-level ``timestamp`` is a naive UTC publish time.

    Verified live 2026-06-10: '2026-06-10 16:46:23' matched current UTC within
    ~20s. Treated as UTC. Falls back to ``fallback`` only if unparseable.
    """

    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value.strip(), fmt).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
    return fallback


def parse_cboe_underlying(payload: dict[str, Any]) -> dict[str, Any]:
    """Underlying market context carried in a CBOE options payload.

    Free extra signal for the committee (price action and 30-day implied vol)
    without another request. Values may be zero/absent off-hours.
    """

    data = payload.get("data") or {}

    def num(key: str) -> float:
        try:
            return float(data.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "price": num("current_price"),
        "change_pct": num("price_change_percent"),
        "prev_close": num("prev_day_close"),
        "day_high": num("high"),
        "day_low": num("low"),
        "volume": int(num("volume")),
        "iv30": num("iv30"),
        "iv30_change": num("iv30_change"),
    }


def parse_cboe_payload(
    payload: dict[str, Any], *, fallback_now: datetime
) -> tuple[datetime, list[dict[str, Any]]]:
    """Normalize a CBOE delayed-quotes payload to (observed_at, contracts).

    Pure (no network) so it is unit-testable from a fixture. Each contract is
    normalized to Moomoo's code plus the fields the liquidity validator needs.
    Unparseable contract symbols are skipped rather than failing the whole chain.
    """

    data = payload.get("data") or {}
    observed_at = _parse_cboe_timestamp(payload.get("timestamp"), fallback_now)
    contracts: list[dict[str, Any]] = []
    for row in data.get("options") or []:
        occ = row.get("option")
        if not occ:
            continue
        try:
            code = occ_to_moomoo_code(str(occ))
            underlying, expiry, side, strike = parse_us_option_code(code)
        except ValueError:
            continue
        contracts.append(
            {
                "code": code,
                "occ": occ,
                "underlying": underlying,
                "expiry": expiry,
                "side": side,
                "strike": strike,
                "bid": float(row.get("bid") or 0.0),
                "ask": float(row.get("ask") or 0.0),
                "open_interest": int(row.get("open_interest") or 0),
                "daily_volume": int(row.get("volume") or 0),
                "iv": float(row.get("iv") or 0.0),
            }
        )
    return observed_at, contracts


class CboeOptionData:
    """Option data from CBOE's free delayed-quotes JSON (no key, ~15 min delay).

    One request returns the full chain for an underlying, so contracts are cached
    briefly per symbol to serve the optionability check, the listing guard, and
    the per-contract quote within a single cycle from one fetch.
    """

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        cache_ttl_seconds: float = 30.0,
        now_fn: Callable[[], datetime] | None = None,
        fetch_fn: Callable[[str], dict[str, Any]] | None = None,
    ):
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._fetch = fetch_fn or self._http_fetch
        # symbol -> (fetched_at, observed_at, contracts, underlying)
        self._cache: dict[
            str, tuple[datetime, datetime, list[dict[str, Any]], dict[str, Any]]
        ] = {}

    def _http_fetch(self, symbol: str) -> dict[str, Any]:
        url = _CBOE_URL.format(symbol=symbol)
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            data = fetch_with_retry(
                url,
                timeout=self.timeout,
                max_retries=2,
                retry_base_delay=1.0,
            )
            return json.loads(data)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise OptionDataError(
                    f"CBOE has no option data for {symbol}"
                ) from exc
            raise OptionDataError(f"CBOE request failed for {symbol}: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OptionDataError(f"CBOE request failed for {symbol}: {exc}") from exc

    def _contracts(self, underlying: str) -> tuple[datetime, list[dict[str, Any]]]:
        symbol = underlying_symbol(underlying)
        now = self._now_fn()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]).total_seconds() < self.cache_ttl_seconds:
            return cached[1], cached[2]
        payload = self._fetch(symbol)
        observed_at, contracts = parse_cboe_payload(payload, fallback_now=now)
        self._cache[symbol] = (now, observed_at, contracts, parse_cboe_underlying(payload))
        return observed_at, contracts

    def underlying_snapshot(self, underlying: str) -> dict[str, Any]:
        """Delayed price/IV context for the underlying, from the cached chain."""

        symbol = underlying_symbol(underlying)
        self._contracts(symbol)
        return self._cache[symbol][3]

    def is_optionable(self, underlying: str) -> bool:
        try:
            _, contracts = self._contracts(underlying)
        except OptionDataError:
            return False
        return bool(contracts)

    def option_expirations(self, underlying: str) -> list[date]:
        _, contracts = self._contracts(underlying)
        return sorted({c["expiry"] for c in contracts})

    def option_chain(
        self, underlying: str, start: date, end: date, option_type: str = "ALL"
    ) -> list[dict[str, Any]]:
        _, contracts = self._contracts(underlying)
        wanted = {"ALL": None, "CALL": "call", "PUT": "put"}.get(option_type.upper())
        rows = [
            c
            for c in contracts
            if start <= c["expiry"] <= end and (wanted is None or c["side"] == wanted)
        ]
        rows.sort(key=lambda c: (c["expiry"], c["side"], c["strike"]))
        return rows

    def is_listed_option(self, option_code: str) -> bool:
        try:
            underlying, _, _, _ = parse_us_option_code(option_code)
            _, contracts = self._contracts(underlying)
        except (ValueError, OptionDataError):
            return False
        return any(c["code"] == option_code for c in contracts)

    def option_quote(
        self,
        *,
        option_code: str,
        expiry: date,
        lot_size: int = US_OPTION_LOT_SIZE,
        now: datetime | None = None,
    ) -> QuoteSnapshot:
        underlying, _, _, _ = parse_us_option_code(option_code)
        observed_at, contracts = self._contracts(underlying)
        match = next((c for c in contracts if c["code"] == option_code), None)
        if match is None:
            raise OptionDataError(
                f"{option_code} not found in CBOE chain for {underlying}"
            )
        if match["ask"] <= 0:
            # No offer means no marketable exit/entry; the validator can't price
            # a zero ask (QuoteSnapshot requires ask > 0) so reject it here.
            raise OptionDataError(f"{option_code} has no ask (no liquidity)")
        return QuoteSnapshot(
            option_code=option_code,
            bid=match["bid"],
            ask=match["ask"],
            open_interest=match["open_interest"],
            daily_volume=match["daily_volume"],
            lot_size=lot_size,
            expiry=expiry,
            observed_at=observed_at,
            is_delayed=True,
        )


def _tradier_contract(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one Tradier chain option row to the shared contract shape."""

    occ = row.get("symbol")
    if not occ:
        return None
    try:
        code = occ_to_moomoo_code(str(occ))
        underlying, expiry, side, strike = parse_us_option_code(code)
    except ValueError:
        return None
    greeks = row.get("greeks") or {}
    return {
        "code": code,
        "occ": occ,
        "underlying": underlying,
        "expiry": expiry,
        "side": side,
        "strike": strike,
        "bid": float(row.get("bid") or 0.0),
        "ask": float(row.get("ask") or 0.0),
        "open_interest": int(row.get("open_interest") or 0),
        "daily_volume": int(row.get("volume") or 0),
        "iv": float(greeks.get("mid_iv") or 0.0),
    }


class TradierOptionData:
    """Fallback option data from the Tradier sandbox REST API.

    Unlike CBOE, Tradier returns one expiration per chain call, so a full
    DTE-window chain requires an expirations call plus one chain call per
    expiration. Used only when CBOE is unavailable, so the extra calls are fine.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = TRADIER_SANDBOX_URL,
        timeout: float = 20.0,
        now_fn: Callable[[], datetime] | None = None,
        fetch_fn: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    ):
        if not token:
            raise ValueError("TradierOptionData requires an API token")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._fetch = fetch_fn or self._http_fetch

    def _http_fetch(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=ssl.create_default_context()
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise OptionDataError(
                f"Tradier request failed ({exc.code}) for {path}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise OptionDataError(f"Tradier request failed for {path}: {exc}") from exc

    def option_expirations(self, underlying: str) -> list[date]:
        payload = self._fetch(
            "/markets/options/expirations",
            {"symbol": underlying_symbol(underlying)},
        )
        node = (payload.get("expirations") or {}).get("date") or []
        if isinstance(node, str):
            node = [node]
        out: list[date] = []
        for value in node:
            try:
                out.append(date.fromisoformat(str(value)))
            except ValueError:
                continue
        return sorted(out)

    def is_optionable(self, underlying: str) -> bool:
        try:
            return bool(self.option_expirations(underlying))
        except OptionDataError:
            return False

    def _chain_for(self, underlying: str, expiry: date) -> list[dict[str, Any]]:
        payload = self._fetch(
            "/markets/options/chains",
            {
                "symbol": underlying_symbol(underlying),
                "expiration": expiry.isoformat(),
                "greeks": "true",
            },
        )
        options = (payload.get("options") or {}).get("option") or []
        if isinstance(options, dict):
            options = [options]
        contracts = []
        for row in options:
            contract = _tradier_contract(row)
            if contract is not None:
                contracts.append(contract)
        return contracts

    def option_chain(
        self, underlying: str, start: date, end: date, option_type: str = "ALL"
    ) -> list[dict[str, Any]]:
        wanted = {"ALL": None, "CALL": "call", "PUT": "put"}.get(option_type.upper())
        rows: list[dict[str, Any]] = []
        for expiry in self.option_expirations(underlying):
            if not start <= expiry <= end:
                continue
            for contract in self._chain_for(underlying, expiry):
                if wanted is None or contract["side"] == wanted:
                    rows.append(contract)
        rows.sort(key=lambda c: (c["expiry"], c["side"], c["strike"]))
        return rows

    def is_listed_option(self, option_code: str) -> bool:
        try:
            underlying, expiry, _, _ = parse_us_option_code(option_code)
            contracts = self._chain_for(underlying, expiry)
        except (ValueError, OptionDataError):
            return False
        return any(c["code"] == option_code for c in contracts)

    def option_quote(
        self,
        *,
        option_code: str,
        expiry: date,
        lot_size: int = US_OPTION_LOT_SIZE,
        now: datetime | None = None,
    ) -> QuoteSnapshot:
        underlying, parsed_expiry, _, _ = parse_us_option_code(option_code)
        contracts = self._chain_for(underlying, parsed_expiry)
        match = next((c for c in contracts if c["code"] == option_code), None)
        if match is None:
            raise OptionDataError(f"{option_code} not found in Tradier chain")
        if match["ask"] <= 0:
            raise OptionDataError(f"{option_code} has no ask (no liquidity)")
        return QuoteSnapshot(
            option_code=option_code,
            bid=match["bid"],
            ask=match["ask"],
            open_interest=match["open_interest"],
            daily_volume=match["daily_volume"],
            lot_size=lot_size,
            expiry=expiry,
            observed_at=now or self._now_fn(),
            is_delayed=True,
        )


class FallbackOptionProvider:
    """Tries each provider in order; a source failure falls through to the next.

    Boolean lookups return True if any source affirms; data lookups return the
    first success and re-raise the last error only if every source fails.
    """

    def __init__(self, providers: list[OptionDataProvider]):
        if not providers:
            raise ValueError("FallbackOptionProvider needs at least one provider")
        self.providers = providers

    def is_optionable(self, underlying: str) -> bool:
        return any(p.is_optionable(underlying) for p in self.providers)

    def is_listed_option(self, option_code: str) -> bool:
        return any(p.is_listed_option(option_code) for p in self.providers)

    def option_expirations(self, underlying: str) -> list[date]:
        return self._first(lambda p: p.option_expirations(underlying))

    def option_chain(
        self, underlying: str, start: date, end: date, option_type: str = "ALL"
    ) -> list[dict[str, Any]]:
        return self._first(lambda p: p.option_chain(underlying, start, end, option_type))

    def option_quote(
        self,
        *,
        option_code: str,
        expiry: date,
        lot_size: int = US_OPTION_LOT_SIZE,
        now: datetime | None = None,
    ) -> QuoteSnapshot:
        return self._first(
            lambda p: p.option_quote(
                option_code=option_code, expiry=expiry, lot_size=lot_size, now=now
            )
        )

    def underlying_snapshot(self, underlying: str) -> dict[str, Any]:
        for provider in self.providers:
            method = getattr(provider, "underlying_snapshot", None)
            if method is None:
                continue
            try:
                return method(underlying)
            except OptionDataError:
                continue
        return {}

    def _first(self, call: Callable[[OptionDataProvider], Any]) -> Any:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return call(provider)
            except OptionDataError as exc:
                last_error = exc
        raise last_error or OptionDataError("no option-data providers available")


def build_option_provider(
    source: str = "yahoo",
    *,
    tradier_token: str = "",
    tradier_base_url: str = TRADIER_SANDBOX_URL,
) -> OptionDataProvider:
    """Assemble the configured option-data provider.

    ``source`` is one of ``yahoo``, ``cboe``, ``tradier``, or combinations
    like ``yahoo+cboe``. Providers are tried in order.
    Recommended: ``yahoo`` (free, reliable).
    """

    from trading_agent.data.yahoo import YahooOptionData

    providers: list[OptionDataProvider] = []
    for name in source.lower().split("+"):
        name = name.strip()
        if name == "yahoo":
            providers.append(YahooOptionData())
        elif name == "cboe":
            providers.append(CboeOptionData())
        elif name == "tradier":
            if tradier_token:
                providers.append(
                    TradierOptionData(tradier_token, base_url=tradier_base_url)
                )
            elif source.lower() == "tradier":
                raise ValueError(
                    "option data source 'tradier' requires TRADING_AGENT_TRADIER_TOKEN"
                )
        elif name:
            raise ValueError(f"unknown option data source: {name!r}")
    if not providers:
        providers.append(YahooOptionData())
    return providers[0] if len(providers) == 1 else FallbackOptionProvider(providers)
