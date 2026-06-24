from trading_agent.execution.orders import (
    OrderManager,
    classify_order_status,
)


class ScriptedBroker:
    def __init__(self, statuses: list[str]):
        self.statuses = statuses
        self.placed = 0
        self.placed_prices: list[float] = []
        self.cancelled: list[str] = []
        self._last_price = 0.21

    def place_limit_order(self, **kwargs):
        self.placed += 1
        self._last_price = kwargs.get("limit_price", 0.21)
        self.placed_prices.append(self._last_price)
        return [{"order_id": f"ord-{self.placed}"}]

    def order_status(self, account_id, order_id, trd_env="SIMULATE"):
        status = self.statuses.pop(0) if self.statuses else "SUBMITTED"
        return {"order_status": status, "dealt_qty": 1, "dealt_avg_price": self._last_price}

    def cancel_order(self, account_id, order_id, trd_env="SIMULATE"):
        self.cancelled.append(order_id)
        return [{"order_status": "CANCELLED_ALL"}]


def _manager(broker, cancel_after=10) -> OrderManager:
    return OrderManager(
        broker=broker,
        account_id=1,
        trd_env="SIMULATE",
        cancel_after_seconds=cancel_after,
        poll_interval_seconds=1,
        sleep_fn=lambda _s: None,
    )


def test_classify_order_status() -> None:
    assert classify_order_status("FILLED_ALL") == "filled"
    assert classify_order_status("FILLED_PART") == "partial"
    assert classify_order_status("SUBMITTED") == "pending"
    assert classify_order_status("CANCELLED_ALL") == "dead"
    assert classify_order_status(None) == "pending"


def test_fills_after_polling() -> None:
    broker = ScriptedBroker(["SUBMITTED", "SUBMITTED", "FILLED_ALL"])
    result = _manager(broker).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy"
    )
    assert result.filled is True
    assert result.dealt_avg_price == 0.21
    assert broker.cancelled == []


def test_order_submitted_callback_fires_after_broker_accepts() -> None:
    broker = ScriptedBroker(["FILLED_ALL"])
    submitted: list[dict] = []

    result = _manager(broker).place_and_await(
        option_code="US.X",
        contracts=1,
        limit_price=0.21,
        side="buy",
        on_order_submitted=submitted.append,
    )

    assert result.filled is True
    assert submitted == [
        {
            "order_id": "ord-1",
            "option_code": "US.X",
            "contracts": 1,
            "limit_price": 0.21,
            "side": "buy",
            "trd_env": "SIMULATE",
        }
    ]


def test_dead_order_returns_unfilled_without_cancel() -> None:
    broker = ScriptedBroker(["FAILED"])
    result = _manager(broker).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy"
    )
    assert result.filled is False
    assert broker.cancelled == []


def test_unfilled_order_is_cancelled_after_timeout() -> None:
    broker = ScriptedBroker(["SUBMITTED"] * 50)
    result = _manager(broker, cancel_after=3).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy"
    )
    assert result.filled is False
    assert result.status == "cancelled_timeout"
    assert broker.cancelled == ["ord-1"]


def test_ladder_fills_at_mid_without_chasing() -> None:
    # Mid rung fills immediately: only one order placed, at the better price.
    broker = ScriptedBroker(["FILLED_ALL"])
    result = _manager(broker).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy",
        price_ladder=[0.20, 0.21],
    )
    assert result.filled is True
    assert result.dealt_avg_price == 0.20
    assert broker.placed_prices == [0.20]
    assert broker.cancelled == []


def test_ladder_chases_to_limit_after_mid_times_out() -> None:
    # Mid never fills (3 polls within its 2s budget) -> cancelled; the second
    # rung at the limit fills on its first poll.
    broker = ScriptedBroker(["SUBMITTED"] * 3 + ["FILLED_ALL"])
    result = _manager(broker, cancel_after=4).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy",
        price_ladder=[0.18, 0.21],
    )
    assert result.filled is True
    assert broker.placed_prices == [0.18, 0.21]
    assert broker.cancelled == ["ord-1"]  # mid rung was cancelled first


def test_ladder_rejected_rung_falls_through_to_next() -> None:
    # Broker rejects the off-tick mid (dead status); the valid limit rung fills.
    broker = ScriptedBroker(["FAILED", "FILLED_ALL"])
    result = _manager(broker).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy",
        price_ladder=[0.18, 0.21],
    )
    assert result.filled is True
    assert broker.placed_prices == [0.18, 0.21]
    assert broker.cancelled == []  # dead orders need no cancel


def test_ladder_unfilled_everywhere_reports_timeout() -> None:
    broker = ScriptedBroker(["SUBMITTED"] * 50)
    result = _manager(broker, cancel_after=4).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="sell",
        price_ladder=[0.20, 0.19],
    )
    assert result.filled is False
    assert result.status == "cancelled_timeout"
    assert broker.placed_prices == [0.20, 0.19]
    assert broker.cancelled == ["ord-1", "ord-2"]


def test_ladder_dedupes_near_equal_rungs() -> None:
    # Mid == limit: a single rung, original single-price behavior.
    broker = ScriptedBroker(["FILLED_ALL"])
    result = _manager(broker).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy",
        price_ladder=[0.21, 0.21],
    )
    assert result.filled is True
    assert broker.placed_prices == [0.21]


def test_missing_order_id_is_unfilled() -> None:
    class NoIdBroker(ScriptedBroker):
        def place_limit_order(self, **kwargs):
            return [{}]

    result = _manager(NoIdBroker([])).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy"
    )
    assert result.filled is False
    assert result.status == "no_order_id"
