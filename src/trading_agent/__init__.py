"""Moomoo MY small-cap options agent foundation."""

import logging

# yfinance logs HTTP 404s at ERROR level when a symbol has no fundamentals
# (e.g. the QQQ/SPY benchmarks have no earnings calendar). Every such call is
# already best-effort and wrapped in try/except returning {}/None, so the
# provider's internal error spam is non-actionable noise -- it flooded the
# paper-loop log with ~44 "No fundamentals data found for symbol: QQQ" lines
# per session. Silence it once here so it is suppressed for every entry point
# (bot, paper loop, CLI). Genuine CRITICALs still surface.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
