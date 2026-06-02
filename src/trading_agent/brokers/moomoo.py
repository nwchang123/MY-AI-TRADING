from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class MoomooBrokerError(RuntimeError):
    """Raised when OpenD or the SDK rejects a broker request."""


@dataclass(frozen=True)
class MoomooConnection:
    host: str
    port: int
    security_firm: str = "FUTUMY"


class MoomooBroker:
    def __init__(self, connection: MoomooConnection):
        self.connection = connection

    def doctor(self) -> dict[str, Any]:
        sdk = self._sdk()
        quote_ctx = sdk.OpenQuoteContext(
            host=self.connection.host,
            port=self.connection.port,
            ai_type=1,
        )
        try:
            ret, data = quote_ctx.get_global_state()
            self._require_ok(sdk, ret, data, "get_global_state")
            return dict(data)
        finally:
            quote_ctx.close()

    def list_us_accounts(self) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        try:
            ret, data = trade_ctx.get_acc_list()
            self._require_ok(sdk, ret, data, "get_acc_list")
            return data.to_dict(orient="records")
        finally:
            trade_ctx.close()

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
        try:
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
        finally:
            trade_ctx.close()

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
        try:
            ret, data = trade_ctx.position_list_query(
                trd_env=self._trd_env(sdk, trd_env), acc_id=account_id
            )
            self._require_ok(sdk, ret, data, "position_list_query")
            return data.to_dict(orient="records")
        finally:
            trade_ctx.close()

    def open_orders_query(
        self, account_id: int, trd_env: str = "SIMULATE"
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        try:
            ret, data = trade_ctx.order_list_query(
                trd_env=self._trd_env(sdk, trd_env), acc_id=account_id
            )
            self._require_ok(sdk, ret, data, "order_list_query")
            return data.to_dict(orient="records")
        finally:
            trade_ctx.close()

    def order_status(
        self, account_id: int, order_id: str, trd_env: str = "SIMULATE"
    ) -> dict[str, Any] | None:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        try:
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
        finally:
            trade_ctx.close()

    def cancel_order(
        self, account_id: int, order_id: str, trd_env: str = "SIMULATE"
    ) -> list[dict[str, Any]]:
        sdk = self._sdk()
        trade_ctx = self._trade_context(sdk)
        try:
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
        finally:
            trade_ctx.close()

    def _trade_context(self, sdk: Any) -> Any:
        try:
            firm = getattr(sdk.SecurityFirm, self.connection.security_firm)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported security firm: {self.connection.security_firm}"
            ) from exc
        return sdk.OpenSecTradeContext(
            filter_trdmarket=sdk.TrdMarket.US,
            host=self.connection.host,
            port=self.connection.port,
            security_firm=firm,
        )

    @staticmethod
    def _sdk() -> Any:
        import moomoo

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

