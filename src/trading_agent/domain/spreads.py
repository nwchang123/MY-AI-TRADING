from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_agent.data.moomoo_market import parse_us_option_code
from trading_agent.domain.proposals import SpreadLeg


class SpreadAnalysisRequest(BaseModel):
    """CLI/file input for deterministic multi-leg payoff analysis."""

    model_config = ConfigDict(extra="forbid")

    legs: list[SpreadLeg] = Field(min_length=2, max_length=4)
    strategy: str | None = None
    underlying_price: float | None = Field(default=None, gt=0)
    price_points: list[float] | None = None

    @field_validator("price_points")
    @classmethod
    def validate_price_points(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("price_points cannot be empty")
        if any(point < 0 for point in value):
            raise ValueError("price_points cannot contain negative prices")
        return value


class SpreadLegAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_code: str
    option_side: str
    action: str
    contracts: int
    strike: float
    expiry: date
    entry_price: float
    entry_cashflow_usd: float


class PayoffPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    underlying_price: float
    pnl_usd: float


class SpreadAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    ticker: str
    expiry: date | None
    legs: list[SpreadLegAnalysis]
    net_debit_usd: float
    net_credit_usd: float
    max_profit_usd: float | None
    max_loss_usd: float | None
    risk_reward: float | None
    breakeven_prices: list[float]
    payoff_points: list[PayoffPoint]
    unlimited_profit: bool
    unlimited_loss: bool


class _ParsedLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leg: SpreadLeg
    ticker: str
    expiry: date
    side: str
    strike: float


def analyze_spread(
    legs: list[SpreadLeg],
    *,
    strategy: str | None = None,
    underlying_price: float | None = None,
    price_points: list[float] | None = None,
    lot_size: int = 100,
) -> SpreadAnalysis:
    """Return deterministic expiration payoff metrics for a 2-4 leg option spread."""

    parsed = [_parse_leg(leg) for leg in legs]
    tickers = {leg.ticker for leg in parsed}
    if len(tickers) != 1:
        raise ValueError("all spread legs must use the same underlying")

    expiries = {leg.expiry for leg in parsed}
    expiry = next(iter(expiries)) if len(expiries) == 1 else None
    entry_cashflow = sum(_entry_cashflow(leg.leg, lot_size) for leg in parsed)
    net_debit = max(0.0, -entry_cashflow)
    net_credit = max(0.0, entry_cashflow)

    high_price = _high_price(parsed, underlying_price, net_debit, lot_size)
    extrema_points = sorted({0.0, high_price, *(leg.strike for leg in parsed)})
    extrema_values = [_expiration_pnl(parsed, price, lot_size) for price in extrema_points]
    high_slope = _high_price_slope(parsed, lot_size)
    unlimited_profit = high_slope > 0
    unlimited_loss = high_slope < 0

    max_profit = None if unlimited_profit else round(max(extrema_values), 4)
    max_loss = None if unlimited_loss else round(abs(min(extrema_values)), 4)
    risk_reward = (
        round(max_profit / max_loss, 4)
        if max_profit is not None and max_loss and max_loss > 0
        else None
    )

    grid = _price_grid(parsed, high_price, underlying_price, price_points)
    inferred_strategy = strategy or classify_spread(legs)
    return SpreadAnalysis(
        strategy=inferred_strategy,
        ticker=next(iter(tickers)),
        expiry=expiry,
        legs=[
            SpreadLegAnalysis(
                option_code=leg.leg.option_code,
                option_side=leg.side,
                action=leg.leg.action,
                contracts=leg.leg.contracts,
                strike=leg.strike,
                expiry=leg.expiry,
                entry_price=leg.leg.limit_price,
                entry_cashflow_usd=round(_entry_cashflow(leg.leg, lot_size), 4),
            )
            for leg in parsed
        ],
        net_debit_usd=round(net_debit, 4),
        net_credit_usd=round(net_credit, 4),
        max_profit_usd=max_profit,
        max_loss_usd=max_loss,
        risk_reward=risk_reward,
        breakeven_prices=_breakevens(parsed, high_price, lot_size),
        payoff_points=[
            PayoffPoint(
                underlying_price=round(price, 4),
                pnl_usd=round(_expiration_pnl(parsed, price, lot_size), 4),
            )
            for price in grid
        ],
        unlimited_profit=unlimited_profit,
        unlimited_loss=unlimited_loss,
    )


def classify_spread(legs: list[SpreadLeg]) -> str:
    """Best-effort strategy label for common 2- and 4-leg option structures."""

    parsed = [_parse_leg(leg) for leg in legs]
    if len(parsed) == 2:
        return _classify_two_leg(parsed)
    if len(parsed) == 4 and _looks_like_iron_condor(parsed):
        return "iron_condor"
    return "custom_spread"


def _parse_leg(leg: SpreadLeg) -> _ParsedLeg:
    ticker, expiry, side, strike = parse_us_option_code(leg.option_code)
    if side != leg.option_side:
        raise ValueError(f"{leg.option_code} side does not match leg.option_side")
    return _ParsedLeg(leg=leg, ticker=ticker, expiry=expiry, side=side, strike=strike)


def _entry_cashflow(leg: SpreadLeg, lot_size: int) -> float:
    signed = 1.0 if leg.action == "sell_to_open" else -1.0
    return signed * leg.limit_price * leg.contracts * lot_size


def _expiration_pnl(legs: list[_ParsedLeg], underlying_price: float, lot_size: int) -> float:
    total = 0.0
    for parsed in legs:
        sign = 1.0 if parsed.leg.action == "buy_to_open" else -1.0
        intrinsic = (
            max(underlying_price - parsed.strike, 0.0)
            if parsed.side == "call"
            else max(parsed.strike - underlying_price, 0.0)
        )
        total += sign * intrinsic * parsed.leg.contracts * lot_size
        total += _entry_cashflow(parsed.leg, lot_size)
    return total


def _high_price_slope(legs: list[_ParsedLeg], lot_size: int) -> float:
    slope = 0.0
    for parsed in legs:
        if parsed.side != "call":
            continue
        sign = 1.0 if parsed.leg.action == "buy_to_open" else -1.0
        slope += sign * parsed.leg.contracts * lot_size
    return round(slope, 8)


def _high_price(
    legs: list[_ParsedLeg],
    underlying_price: float | None,
    net_debit_usd: float,
    lot_size: int,
) -> float:
    max_strike = max(leg.strike for leg in legs)
    debit_per_share = net_debit_usd / lot_size if lot_size > 0 else 0.0
    anchors = [max_strike * 2.0, max_strike + debit_per_share * 3.0 + 1.0]
    if underlying_price is not None:
        anchors.append(underlying_price * 2.0)
    return max(anchors)


def _price_grid(
    legs: list[_ParsedLeg],
    high_price: float,
    underlying_price: float | None,
    price_points: list[float] | None,
) -> list[float]:
    if price_points is not None:
        return sorted(set(float(point) for point in price_points))
    anchors = {0.0, high_price, *(leg.strike for leg in legs)}
    if underlying_price is not None:
        anchors.add(underlying_price)
    step = high_price / 20.0 if high_price > 0 else 1.0
    anchors.update(round(step * idx, 6) for idx in range(21))
    return sorted(anchors)


def _breakevens(legs: list[_ParsedLeg], high_price: float, lot_size: int) -> list[float]:
    points = sorted({0.0, high_price, *(leg.strike for leg in legs)})
    roots: list[float] = []
    previous_x = points[0]
    previous_y = _expiration_pnl(legs, previous_x, lot_size)
    if abs(previous_y) < 1e-9:
        roots.append(previous_x)
    for current_x in points[1:]:
        current_y = _expiration_pnl(legs, current_x, lot_size)
        if abs(current_y) < 1e-9:
            roots.append(current_x)
        elif previous_y * current_y < 0:
            span = current_x - previous_x
            roots.append(previous_x - previous_y * span / (current_y - previous_y))
        previous_x, previous_y = current_x, current_y
    return _unique_rounded(roots)


def _unique_rounded(values: list[float]) -> list[float]:
    rounded = sorted(round(value, 4) for value in values)
    unique: list[float] = []
    for value in rounded:
        if not unique or abs(value - unique[-1]) > 1e-4:
            unique.append(value)
    return unique


def _classify_two_leg(legs: list[_ParsedLeg]) -> str:
    first, second = sorted(legs, key=lambda item: item.strike)
    same_expiry = first.expiry == second.expiry
    if first.side == second.side and same_expiry:
        if first.side == "call":
            if first.leg.action == "buy_to_open" and second.leg.action == "sell_to_open":
                return "bull_call_spread"
            if first.leg.action == "sell_to_open" and second.leg.action == "buy_to_open":
                return "bear_call_credit_spread"
        if first.side == "put":
            if first.leg.action == "sell_to_open" and second.leg.action == "buy_to_open":
                return "bear_put_spread"
            if first.leg.action == "buy_to_open" and second.leg.action == "sell_to_open":
                return "bull_put_credit_spread"

    actions = {leg.leg.action for leg in legs}
    sides = {leg.side for leg in legs}
    if sides == {"call", "put"} and same_expiry:
        if actions == {"buy_to_open"}:
            return "long_straddle" if first.strike == second.strike else "long_strangle"
        if actions == {"sell_to_open"}:
            return "short_straddle" if first.strike == second.strike else "short_strangle"
    return "custom_spread"


def _looks_like_iron_condor(legs: list[_ParsedLeg]) -> bool:
    expiries = {leg.expiry for leg in legs}
    sides = [leg.side for leg in legs]
    if len(expiries) != 1 or sides.count("put") != 2 or sides.count("call") != 2:
        return False
    puts = sorted([leg for leg in legs if leg.side == "put"], key=lambda item: item.strike)
    calls = sorted([leg for leg in legs if leg.side == "call"], key=lambda item: item.strike)
    return (
        puts[0].leg.action == "buy_to_open"
        and puts[1].leg.action == "sell_to_open"
        and calls[0].leg.action == "sell_to_open"
        and calls[1].leg.action == "buy_to_open"
    )
