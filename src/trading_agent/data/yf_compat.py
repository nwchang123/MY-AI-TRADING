from __future__ import annotations

import warnings
from typing import Any

_UTCNOW_MESSAGE = r"Timestamp\.utcnow is deprecated"


def import_yfinance() -> Any:
    """Import yfinance with the pandas ``Timestamp.utcnow`` deprecation silenced.

    yfinance's quote scraper (``.info``/``.calendar``) still calls the
    pandas-deprecated ``Timestamp.utcnow()`` on every fetch. The process-wide
    ignore in :mod:`trading_agent` (``__init__``) is set at package import, but
    importing yfinance *afterwards* PREPENDS yfinance's own warning filters above
    ours, burying it -- so the ``Pandas4Warning`` still leaked ~1196x in one
    overnight session (≈1/5 of the loop log). Re-applying the ignore right after
    the import re-hoists it to the front of ``warnings.filters``; yfinance is
    cached after the first import, so this stays in front for the rest of the
    process and covers every later ``.info``/``.calendar``/``.history`` call.
    """
    import yfinance as yf

    warnings.filterwarnings("ignore", message=_UTCNOW_MESSAGE)
    return yf
