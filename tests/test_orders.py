from trading_agent.execution.orders import (
    OrderManager,
    classify_order_status,
)


class ScriptedBroker:
    def __init__(self, statuses: list[str]):
        self.statuses = statuses
        self.placed = 0
        self.cancelled: list[str] = []

    def place_limit_order(self, **kwargs):
        self.placed += 1
        return [{"order_id": "ord-1"}]

    def order_status(self, account_id, order_id, trd_env="SIMULATE"):
        status = self.statuses.pop(0) if self.statuses else "SUBMITTED"
        return {"order_status": status, "dealt_qty": 1, "dealt_avg_price": 0.21}

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


def test_missing_order_id_is_unfilled() -> None:
    class NoIdBroker(ScriptedBroker):
        def place_limit_order(self, **kwargs):
            return [{}]

    result = _manager(NoIdBroker([])).place_and_await(
        option_code="US.X", contracts=1, limit_price=0.21, side="buy"
    )
    assert result.filled is False
    assert result.status == "no_order_id"
