from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

OrderClass = Literal["filled", "partial", "pending", "dead"]

# Moomoo OrderStatus strings (Trd_Common). Single-contract orders cannot fill
# partially, so partial is treated as still pending.
_FILLED = {"FILLED_ALL"}
_PARTIAL = {"FILLED_PART"}
_DEAD = {
    "CANCELLED_ALL",
    "CANCELLED_PART",
    "FAILED",
    "DISABLED",
    "DELETED",
    "SUBMIT_FAILED",
    "TIMEOUT",
}


def classify_order_status(status: str | None) -> OrderClass:
    value = (status or "").strip().upper()
    if value in _FILLED:
        return "filled"
    if value in _PARTIAL:
        return "partial"
    if value in _DEAD:
        return "dead"
    return "pending"


@dataclass
class FillResult:
    filled: bool
    status: str
    order_id: str | None = None
    dealt_qty: float = 0.0
    dealt_avg_price: float = 0.0


class OrderManager:
    """Places a limit order and confirms the actual fill.

    Polls order status until the order fills or dies, and cancels it if it is
    still unfilled after ``cancel_after_seconds``. This closes the gap where the
    cycle assumed an order filled at its limit price; the position is only
    recorded on a confirmed fill, at the real dealt price.
    """

    def __init__(
        self,
        *,
        broker: Any,
        account_id: int,
        trd_env: str,
        cancel_after_seconds: int,
        poll_interval_seconds: float = 2.0,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.broker = broker
        self.account_id = account_id
        self.trd_env = trd_env
        self.cancel_after_seconds = cancel_after_seconds
        self.poll_interval_seconds = max(0.1, poll_interval_seconds)
        self.sleep_fn = sleep_fn or time.sleep

    def place_and_await(
        self,
        *,
        option_code: str,
        contracts: int,
        limit_price: float,
        side: str,
        price_ladder: list[float] | None = None,
    ) -> FillResult:
        """Work a limit order through a price ladder until it fills.

        ``price_ladder`` lists prices from best-for-us to most-aggressive (a
        buy climbs toward the ask, a sell descends toward the bid); each rung
        gets an equal share of ``cancel_after_seconds`` and is cancelled
        before the next is placed, so at most one order is ever live. A rung
        the broker rejects (e.g. an off-tick price) falls through to the next
        rung instead of aborting -- the final rung is always the caller's
        quote-derived ``limit_price``, which is a valid tick. Without a
        ladder, behavior is the original single-price place-poll-cancel.
        """

        rungs: list[float] = []
        for price in price_ladder or [limit_price]:
            price = round(float(price), 2)
            if price > 0 and (not rungs or abs(price - rungs[-1]) >= 0.005):
                rungs.append(price)
        if not rungs:
            rungs = [round(limit_price, 2)]
        budget = self.cancel_after_seconds / len(rungs)

        result = FillResult(filled=False, status="no_order_id")
        for price in rungs:
            result = self._place_one_rung(
                option_code=option_code,
                contracts=contracts,
                limit_price=price,
                side=side,
                budget_seconds=budget,
            )
            if result.filled:
                return result
        return result

    def _place_one_rung(
        self,
        *,
        option_code: str,
        contracts: int,
        limit_price: float,
        side: str,
        budget_seconds: float,
    ) -> FillResult:
        record = self.broker.place_limit_order(
            account_id=self.account_id,
            option_code=option_code,
            contracts=contracts,
            limit_price=limit_price,
            side=side,
            trd_env=self.trd_env,
        )
        order_id = self._order_id(record)
        if order_id is None:
            return FillResult(filled=False, status="no_order_id")

        elapsed = 0.0
        while True:
            status_row = self.broker.order_status(
                self.account_id, order_id, self.trd_env
            )
            status = str((status_row or {}).get("order_status", "UNKNOWN"))
            classification = classify_order_status(status)
            if classification == "filled":
                return FillResult(
                    filled=True,
                    status=status,
                    order_id=order_id,
                    dealt_qty=float((status_row or {}).get("dealt_qty") or contracts),
                    dealt_avg_price=float(
                        (status_row or {}).get("dealt_avg_price") or limit_price
                    ),
                )
            if classification == "dead":
                return FillResult(filled=False, status=status, order_id=order_id)
            if elapsed >= budget_seconds:
                break
            self.sleep_fn(self.poll_interval_seconds)
            elapsed += self.poll_interval_seconds

        self.broker.cancel_order(self.account_id, order_id, self.trd_env)
        return FillResult(filled=False, status="cancelled_timeout", order_id=order_id)

    @staticmethod
    def _order_id(record: Any) -> str | None:
        if isinstance(record, list) and record:
            value = record[0].get("order_id")
            return str(value) if value is not None else None
        return None
