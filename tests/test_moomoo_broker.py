from types import SimpleNamespace

import pytest

from trading_agent.brokers.moomoo import (
    MoomooBroker,
    MoomooBrokerError,
    MoomooConnection,
    assert_opend_reachable,
)


def test_preflight_raises_on_dead_gateway() -> None:
    # Nothing listens on this port: the preflight must fail fast (the SDK
    # itself would retry a refused connection forever).
    with pytest.raises(MoomooBrokerError, match="not reachable"):
        assert_opend_reachable("127.0.0.1", 1, timeout=0.5)


def test_preflight_passes_with_listener() -> None:
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        assert_opend_reachable("127.0.0.1", server.getsockname()[1], timeout=1.0)
    finally:
        server.close()


class FakeFrame:
    def __init__(self, records: list[dict[str, object]]):
        self.records = records

    def to_dict(self, *, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return self.records


class FakeTradeContext:
    def __init__(self):
        self.closed = False
        self.place_order_kwargs: dict[str, object] | None = None

    def place_order(self, **kwargs):
        self.place_order_kwargs = kwargs
        return 0, FakeFrame([{"order_id": "paper-1"}])

    def close(self) -> None:
        self.closed = True


def _fake_sdk(trade_ctx: FakeTradeContext):
    constructor_kwargs: dict[str, object] = {}

    def open_trade_context(**kwargs):
        constructor_kwargs.update(kwargs)
        return trade_ctx

    return (
        SimpleNamespace(
            RET_OK=0,
            OpenSecTradeContext=open_trade_context,
            SecurityFirm=SimpleNamespace(FUTUMY="FUTUMY"),
            TrdMarket=SimpleNamespace(US="US"),
            TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
            OrderType=SimpleNamespace(NORMAL="NORMAL"),
            TrdEnv=SimpleNamespace(SIMULATE="SIMULATE", REAL="REAL"),
            TimeInForce=SimpleNamespace(DAY="DAY"),
        ),
        constructor_kwargs,
    )


def test_paper_order_is_forced_to_simulate_and_closes_context(monkeypatch) -> None:
    trade_ctx = FakeTradeContext()
    sdk, constructor_kwargs = _fake_sdk(trade_ctx)
    monkeypatch.setattr(MoomooBroker, "_sdk", staticmethod(lambda: sdk))
    broker = MoomooBroker(MoomooConnection(host="127.0.0.1", port=11111))

    records = broker.place_paper_limit_order(
        account_id=123,
        option_code="US.EXAMPLE260626C00005000",
        contracts=1,
        limit_price=0.2,
        side="buy",
    )

    assert records == [{"order_id": "paper-1"}]
    assert constructor_kwargs == {
        "filter_trdmarket": "US",
        "host": "127.0.0.1",
        "port": 11111,
        "security_firm": "FUTUMY",
    }
    assert trade_ctx.place_order_kwargs == {
        "price": 0.2,
        "qty": 1,
        "code": "US.EXAMPLE260626C00005000",
        "trd_side": "BUY",
        "order_type": "NORMAL",
        "trd_env": "SIMULATE",
        "acc_id": 123,
        "time_in_force": "DAY",
    }
    assert trade_ctx.closed is True


def test_live_order_passes_real_env(monkeypatch) -> None:
    trade_ctx = FakeTradeContext()
    sdk, _ = _fake_sdk(trade_ctx)
    monkeypatch.setattr(MoomooBroker, "_sdk", staticmethod(lambda: sdk))
    broker = MoomooBroker(MoomooConnection(host="127.0.0.1", port=11111))

    broker.place_limit_order(
        account_id=555,
        option_code="US.EXAMPLE260626C00005000",
        contracts=1,
        limit_price=0.2,
        side="buy",
        trd_env="REAL",
    )

    assert trade_ctx.place_order_kwargs["trd_env"] == "REAL"


def test_invalid_trd_env_is_rejected(monkeypatch) -> None:
    trade_ctx = FakeTradeContext()
    sdk, _ = _fake_sdk(trade_ctx)
    monkeypatch.setattr(MoomooBroker, "_sdk", staticmethod(lambda: sdk))
    broker = MoomooBroker(MoomooConnection(host="127.0.0.1", port=11111))

    with pytest.raises(ValueError, match="trd_env"):
        broker.place_limit_order(
            account_id=555,
            option_code="US.EXAMPLE260626C00005000",
            contracts=1,
            limit_price=0.2,
            side="buy",
            trd_env="PAPER",
        )


def test_invalid_paper_order_side_is_rejected_before_context_creation(monkeypatch) -> None:
    trade_ctx = FakeTradeContext()
    sdk, constructor_kwargs = _fake_sdk(trade_ctx)
    monkeypatch.setattr(MoomooBroker, "_sdk", staticmethod(lambda: sdk))
    broker = MoomooBroker(MoomooConnection(host="127.0.0.1", port=11111))

    with pytest.raises(ValueError, match="side must be"):
        broker.place_paper_limit_order(
            account_id=123,
            option_code="US.EXAMPLE260626C00005000",
            contracts=1,
            limit_price=0.2,
            side="hold",
        )

    assert constructor_kwargs == {}
