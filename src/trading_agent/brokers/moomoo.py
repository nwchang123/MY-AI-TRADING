from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_agent.data.moomoo_logging import quiet_moomoo_console_logs


class MoomooBrokerError(RuntimeError):
    """Raised when OpenD or the SDK rejects a broker request."""


def assert_opend_reachable(host: str, port: int, timeout: float = 3.0) -> None:
    """Fast TCP preflight for the OpenD gateway.

    The moomoo SDK retries a refused connection indefinitely instead of
    raising (observed live: 300+ retries, cycle hung, no alert). Checking the
    socket first turns a dead gateway into an immediate, alertable error and
    lets the loop keep ticking until OpenD comes back.
    """

    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise MoomooBrokerError(
            f"OpenD gateway is not reachable on {host}:{port} -- "
            "start/log in the OpenD GUI"
        ) from exc


@dataclass(frozen=True)
class MoomooConnection:
    host: str
    port: int
    security_firm: str = "FUTUMY"


class MoomooBroker:
    def __init__(self, connection: MoomooConnection):
        self.connection = connection
        self._trade_ctx: Any | None = None
        self._quote_ctx: Any | None = None

    def doctor(self) -> dict[str, Any]:
        sdk = self._sdk()
        quote_ctx = self._quote_context(sdk)
        ret, data = quote_ctx.get_global_state()
        self._require_ok(sdk, ret, data, "get_global_state")
        return dict(data)

    def list_us_accounts(self) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        ret, data = trade_ctx.get_acc_list()
        self._require_ok(sdk, ret, data, "get_acc_list")
        return data.to_dict(orient="records")

    def place_limit_order(
        self,
        *,
        account_id: int,
        option_code: str,
        contracts: int,
        limit_price: float,
        side: str,
        trd_env: str = "SIMULATE",
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if trd_env not in {"SIMULATE", "REAL"}:
            raise ValueError("trd_env must be 'SIMULATE' or 'REAL'")
        env = sdk.TrdEnv.SIMULATE if trd_env == "SIMULATE" else sdk.TrdEnv.REAL
        trade_ctx = self._trade_context(sdk)
        ret, data = trade_ctx.place_order(
            price=limit_price,
            qty=contracts,
            code=option_code,
            trd_side=sdk.TrdSide.BUY if side == "buy" else sdk.TrdSide.SELL,
            order_type=sdk.OrderType.NORMAL,
            trd_env=env,
            acc_id=account_id,
            time_in_force=sdk.TimeInForce.DAY,
        )
        self._require_ok(sdk, ret, data, "place_order")
        return data.to_dict(orient="records")

    def place_paper_limit_order(
        self,
        *,
        account_id: int,
        option_code: str,
        contracts: int,
        limit_price: float,
        side: str,
    ) -> list[dict[str, Any]]:
        return self.place_limit_order(
            account_id=account_id,
            option_code=option_code,
            contracts=contracts,
            limit_price=limit_price,
            side=side,
            trd_env="SIMULATE",
        )

    def positions_query(
        self, account_id: int, trd_env: str = "SIMULATE"
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        ret, data = trade_ctx.position_list_query(
            trd_env=self._trd_env(sdk, trd_env), acc_id=account_id
        )
        self._require_ok(sdk, ret, data, "position_list_query")
        return data.to_dict(orient="records")

    def open_orders_query(
        self, account_id: int, trd_env: str = "SIMULATE"
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        ret, data = trade_ctx.order_list_query(
            trd_env=self._trd_env(sdk, trd_env), acc_id=account_id
        )
        self._require_ok(sdk, ret, data, "order_list_query")
        return data.to_dict(orient="records")

    def order_status(
        self, account_id: int, order_id: str, trd_env: str = "SIMULATE"
    ) -> dict[str, Any] | None:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        ret, data = trade_ctx.order_list_query(
            order_id=order_id,
            trd_env=self._trd_env(sdk, trd_env),
            acc_id=account_id,
        )
        self._require_ok(sdk, ret, data, "order_list_query")
        records = data.to_dict(orient="records")
        for record in records:
            if str(record.get("order_id")) == str(order_id):
                return record
        return records[0] if records else None

    def cancel_order(
        self, account_id: int, order_id: str, trd_env: str = "SIMULATE"
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        ret, data = trade_ctx.modify_order(
            modify_order_op=sdk.ModifyOrderOp.CANCEL,
            order_id=order_id,
            qty=0,
            price=0,
            trd_env=self._trd_env(sdk, trd_env),
            acc_id=account_id,
        )
        self._require_ok(sdk, ret, data, "modify_order")
        return data.to_dict(orient="records")

    def _trade_context(self, sdk: Any) -> Any:
        if self._trade_ctx is not None:
            return self._trade_ctx
        try:
            firm = getattr(sdk.SecurityFirm, self.connection.security_firm)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported security firm: {self.connection.security_firm}"
            ) from exc
        self._trade_ctx = sdk.OpenSecTradeContext(
            filter_trdmarket=sdk.TrdMarket.US,
            host=self.connection.host,
            port=self.connection.port,
            security_firm=firm,
        )
        return self._trade_ctx

    def _quote_context(self, sdk: Any) -> Any:
        if self._quote_ctx is None:
            self._quote_ctx = sdk.OpenQuoteContext(
                host=self.connection.host,
                port=self.connection.port,
                ai_type=1,
            )
        return self._quote_ctx

    def close(self) -> None:
        for attr in ("_trade_ctx", "_quote_ctx"):
            ctx = getattr(self, attr)
            if ctx is None:
                continue
            try:
                ctx.close()
            finally:
                setattr(self, attr, None)

    @staticmethod
    def _sdk() -> Any:
        import moomoo

        quiet_moomoo_console_logs(moomoo)
        return moomoo

    @staticmethod
    def _trd_env(sdk: Any, trd_env: str) -> Any:
        if trd_env == "SIMULATE":
            return sdk.TrdEnv.SIMULATE
        if trd_env == "REAL":
            return sdk.TrdEnv.REAL
        raise ValueError("trd_env must be 'SIMULATE' or 'REAL'")

    @staticmethod
    def _require_ok(sdk: Any, ret: int, data: Any, operation: str) -> None:
        if ret != sdk.RET_OK:
            raise MoomooBrokerError(f"{operation} failed: {data}")
