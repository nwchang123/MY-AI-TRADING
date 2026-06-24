"""Tame the moomoo SDK's console logging.

Kept dependency-free (only stdlib) so both the market adapter and the broker
can import it without the moomoo_market <-> brokers.moomoo import cycle.
"""
from __future__ import annotations

import logging
from typing import Any


def quiet_moomoo_console_logs(moomoo_module: Any) -> None:
    """Silence the SDK's WARNING-level connect/disconnect console spam.

    The moomoo SDK logs every per-request connect/disconnect to stdout at
    WARNING level via its FTLog console handler (~2,000 lines/session flooded
    the paper loop). Raise the console threshold to ERROR so real failures
    still surface while the routine churn is dropped; the SDK's own rotating
    file log under %APPDATA% keeps the full record. Idempotent and defensive --
    logging configuration must never break trading.
    """
    try:
        moomoo_module.logger.console_level = logging.ERROR
    except Exception:  # noqa: BLE001 - never let logging config break trading
        pass
