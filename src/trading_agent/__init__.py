"""Moomoo MY small-cap options agent foundation."""

import logging
import warnings

# yfinance logs HTTP 404s at ERROR level when a symbol has no fundamentals
# (e.g. the QQQ/SPY benchmarks have no earnings calendar). Every such call is
# already best-effort and wrapped in try/except returning {}/None, so the
# provider's internal error spam is non-actionable noise -- it flooded the
# paper-loop log with ~44 "No fundamentals data found for symbol: QQQ" lines
# per session. Silence it once here so it is suppressed for every entry point
# (bot, paper loop, CLI). Genuine CRITICALs still surface.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# yfinance still calls the pandas-deprecated Timestamp.utcnow() on every quote
# scrape, emitting a Pandas4Warning we cannot fix upstream. It fired ~5,858
# times in one session (≈1/3 of the loop log). Filter it by message so the
# noise is gone regardless of the warning's category; unrelated warnings are
# untouched. NOTE: importing yfinance later PREPENDS its own filters above this
# one, burying it (the warning still leaked ~1196x on 2026-06-26) -- so
# data.yf_compat.import_yfinance() RE-applies this filter right after the import
# to re-hoist it. This line still covers the window before yfinance is imported.
warnings.filterwarnings(
    "ignore", message=r"Timestamp\.utcnow is deprecated"
)
