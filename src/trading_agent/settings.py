from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Mode = Literal["paper", "live"]
LIVE_ACKNOWLEDGMENT = "I_UNDERSTAND_REAL_MONEY"


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    mode: Mode
    moomoo_host: str
    moomoo_port: int
    security_firm: str
    account_id: int | None
    live_acknowledgment: str
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_model_pro: str = "deepseek-v4-pro"
    # Optional cross-provider model for the skeptic / risk_manager roles. When
    # set, the adversarial roles run on a different model lineage so their errors
    # are uncorrelated with the primary model. Base URL / key fall back to the
    # primary endpoint when left blank.
    llm_adversary_api_key: str = ""
    llm_adversary_base_url: str = ""
    llm_adversary_model: str = ""
    sec_user_agent: str = ""
    live_account_allowlist: tuple[int, ...] = ()
    # Option data comes from a free delayed feed (Moomoo does not entitle US
    # option quotes); execution stays on Moomoo. Source is one of yahoo | cboe |
    # tradier, or a '+'-joined fallback chain like yahoo+cboe (tried in order).
    option_data_source: str = "cboe"
    tradier_token: str = ""
    tradier_base_url: str = "https://sandbox.tradier.com/v1"
    # Finnhub free key for the bulk upcoming-earnings calendar (pre-catalyst
    # selection). Blank -> the earnings calendar falls back to a yfinance
    # per-ticker shortlist.
    finnhub_api_key: str = ""
    # Operator alerts (optional): Telegram bot token + the operator's chat id.
    # Blank disables alerting; a notifier failure never blocks a cycle.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "Settings":
        root = (root_dir or Path.cwd()).resolve()
        load_dotenv(root / ".env")
        mode = os.getenv("TRADING_AGENT_MODE", "paper").strip().lower()
        if mode not in {"paper", "live"}:
            raise ValueError("TRADING_AGENT_MODE must be 'paper' or 'live'")

        raw_account_id = os.getenv("TRADING_AGENT_ACCOUNT_ID", "").strip()
        raw_allowlist = os.getenv("TRADING_AGENT_LIVE_ACCOUNT_ALLOWLIST", "").strip()
        allowlist = tuple(
            int(part.strip()) for part in raw_allowlist.split(",") if part.strip()
        )
        return cls(
            root_dir=root,
            mode=mode,  # type: ignore[arg-type]
            moomoo_host=os.getenv("TRADING_AGENT_MOOMOO_HOST", "127.0.0.1"),
            moomoo_port=int(os.getenv("TRADING_AGENT_MOOMOO_PORT", "11111")),
            security_firm=os.getenv("TRADING_AGENT_SECURITY_FIRM", "FUTUMY").strip(),
            account_id=int(raw_account_id) if raw_account_id else None,
            live_acknowledgment=os.getenv("TRADING_AGENT_ENABLE_LIVE", "").strip(),
            llm_api_key=os.getenv("TRADING_AGENT_LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv(
                "TRADING_AGENT_LLM_BASE_URL", "https://api.deepseek.com"
            ).strip(),
            llm_model=os.getenv("TRADING_AGENT_LLM_MODEL", "deepseek-v4-flash").strip(),
            llm_model_pro=os.getenv(
                "TRADING_AGENT_LLM_MODEL_PRO", "deepseek-v4-pro"
            ).strip(),
            llm_adversary_api_key=os.getenv(
                "TRADING_AGENT_LLM_ADVERSARY_API_KEY", ""
            ).strip(),
            llm_adversary_base_url=os.getenv(
                "TRADING_AGENT_LLM_ADVERSARY_BASE_URL", ""
            ).strip(),
            llm_adversary_model=os.getenv(
                "TRADING_AGENT_LLM_ADVERSARY_MODEL", ""
            ).strip(),
            sec_user_agent=os.getenv("TRADING_AGENT_SEC_USER_AGENT", "").strip(),
            live_account_allowlist=allowlist,
            option_data_source=os.getenv(
                "TRADING_AGENT_OPTION_DATA_SOURCE", "cboe"
            ).strip(),
            tradier_token=os.getenv("TRADING_AGENT_TRADIER_TOKEN", "").strip(),
            tradier_base_url=os.getenv(
                "TRADING_AGENT_TRADIER_BASE_URL", "https://sandbox.tradier.com/v1"
            ).strip(),
            finnhub_api_key=os.getenv("TRADING_AGENT_FINNHUB_API_KEY", "").strip(),
            telegram_bot_token=os.getenv(
                "TRADING_AGENT_TELEGRAM_BOT_TOKEN", ""
            ).strip(),
            telegram_chat_id=os.getenv("TRADING_AGENT_TELEGRAM_CHAT_ID", "").strip(),
        )

    @property
    def mandate_path(self) -> Path:
        return self.root_dir / "config" / f"mandate.{self.mode}.yaml"

    def assert_live_startup_allowed(self) -> None:
        if self.mode != "live":
            return
        if self.live_acknowledgment != LIVE_ACKNOWLEDGMENT:
            raise RuntimeError(
                "Live mode is disabled. Set TRADING_AGENT_ENABLE_LIVE="
                f"{LIVE_ACKNOWLEDGMENT} manually for a controlled live run."
            )
        if self.account_id is None:
            raise RuntimeError("TRADING_AGENT_ACCOUNT_ID is required in live mode.")
        if not self.live_account_allowlist:
            raise RuntimeError(
                "Live mode requires an operator-pinned allowlist. Set "
                "TRADING_AGENT_LIVE_ACCOUNT_ALLOWLIST to the approved account id(s)."
            )
        if self.account_id not in self.live_account_allowlist:
            raise RuntimeError(
                f"Account {self.account_id} is not in the live allowlist "
                f"{self.live_account_allowlist}."
            )

