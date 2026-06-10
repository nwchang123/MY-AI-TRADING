from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal[
    "sec_8k",
    "sec_10q",
    "sec_10k",
    "sec_s3",
    "sec_424b",
    "sec_13d",
    "sec_13g",
    "sec_form4",
    "press_release",
    "investor_relations",
    "moomoo_news",
    "news_rss",
    "earnings_calendar",
    "regulatory_calendar",
    "options_flow",
]


class EvidenceItem(BaseModel):
    """A single timestamped public-data observation behind a candidate."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    source_type: SourceType
    source_url: str = Field(min_length=1)
    published_at: datetime
    observed_fact: str = Field(min_length=1)
    retrieved_at: datetime


class CandidateContext(BaseModel):
    """Everything the committee is allowed to reason over for one ticker."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    as_of: datetime
    evidence: list[EvidenceItem] = Field(min_length=1)

    def evidence_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence}
