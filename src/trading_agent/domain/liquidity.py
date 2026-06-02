from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from trading_agent.domain.risk import ExecutionMandate, OptionsMandate, QuoteSnapshot


class LiquidityResult(BaseModel):
    """Deterministic pre-trade verdict for one option contract.

    Runs at scan time on the contract itself, before any LLM proposal exists. It
    mirrors the relevant mandate checks so unusable contracts never reach the
    committee, and computes the marketable limit price and its allowed ceiling.
    """

    model_config = ConfigDict(extra="forbid")

    option_code: str
    passed: bool
    reasons: list[str]
    spread_pct: float
    dte: int
    estimated_contract_cost_usd: float
    suggested_limit_price: float
    max_limit_price: float


class LiquidityValidator:
    """Validates a live option quote against the deterministic mandate rules."""

    def __init__(self, options: OptionsMandate, execution: ExecutionMandate):
        self.options = options
        self.execution = execution

    def validate(
        self,
        quote: QuoteSnapshot,
        contracts: int = 1,
        now: datetime | None = None,
    ) -> LiquidityResult:
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        reasons: list[str] = []
        spread_pct = self._spread_pct(quote.bid, quote.ask)
        # Worst-case fill is the ask, so cost and the suggested limit use it.
        estimated_cost = (
            quote.ask * quote.lot_size * contracts + self.options.fee_buffer_usd
        )
        max_limit_price = round(
            quote.ask * (1 + self.execution.max_limit_chase_pct / 100), 4
        )

        if quote.bid < self.options.min_option_bid_usd:
            reasons.append("option bid is below minimum")
        if quote.ask <= quote.bid:
            reasons.append("market is crossed or locked")
        if spread_pct > self.options.max_bid_ask_spread_pct:
            reasons.append("bid-ask spread is too wide")
        if quote.open_interest < self.options.min_open_interest:
            reasons.append("open interest is below minimum")
        if quote.daily_volume < self.options.min_daily_volume:
            reasons.append("daily option volume is below minimum")

        quote_age = (current_time - quote.observed_at).total_seconds()
        if quote_age < 0 or quote_age > self.execution.stale_quote_seconds:
            reasons.append("quote is stale")

        dte = (quote.expiry - current_time.date()).days
        if not self.options.min_dte <= dte <= self.options.max_dte:
            reasons.append("days to expiry violate mandate")

        if estimated_cost > self.options.max_contract_cost_usd:
            reasons.append("contract cost exceeds mandate")

        return LiquidityResult(
            option_code=quote.option_code,
            passed=not reasons,
            reasons=reasons,
            spread_pct=round(spread_pct, 4),
            dte=dte,
            estimated_contract_cost_usd=round(estimated_cost, 4),
            suggested_limit_price=round(quote.ask, 4),
            max_limit_price=max_limit_price,
        )

    @staticmethod
    def _spread_pct(bid: float, ask: float) -> float:
        if ask <= bid or bid <= 0:
            return float("inf")
        return ((ask - bid) / ((ask + bid) / 2)) * 100
