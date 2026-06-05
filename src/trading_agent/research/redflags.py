from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trading_agent.domain.evidence import CandidateContext, EvidenceItem, SourceType
from trading_agent.research.catalysts import (
    DILUTION_KEYWORDS,
    DILUTION_TYPES,
    INSIDER_SELL_KEYWORDS,
)
from trading_agent.research.scoring import ScoreComponents

# Tunable thresholds. A dilution registration or an insider sale only counts as
# a hard (vetoing) red flag while it is still fresh; older items downgrade to a
# warning the committee weighs but is not blocked by. These are deliberately
# explicit constants so the operator can dial how strict the deterministic
# filter is.
DILUTION_FRESH_DAYS = 45.0
INSIDER_SELL_FRESH_DAYS = 30.0
STALE_CATALYST_DAYS = 30.0
# Weighted catalyst points below which the deterministic catalyst signal is
# considered weak (CATALYST_WEIGHT is 30, so ~0.33 of full strength).
WEAK_CATALYST_SCORE = 10.0

RedFlagSeverity = Literal["warn", "critical"]

# Source types that corroborate a thesis with a primary / official record.
_PRIMARY_SOURCES: set[SourceType] = {
    "sec_8k",
    "sec_10q",
    "sec_10k",
    "sec_13d",
    "sec_13g",
    "regulatory_calendar",
    "earnings_calendar",
    "investor_relations",
}


class RedFlag(BaseModel):
    """One deterministic, code-detected concern about a candidate.

    ``critical`` flags are hard disqualifiers (a known dilution overhang or
    insider selling) that block the trade without consulting the LLM. ``warn``
    flags are handed to the skeptic and risk_manager as confirmed facts to weigh.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: RedFlagSeverity
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


def _age_days(item: EvidenceItem, as_of: datetime) -> float:
    return max(0.0, (as_of - item.published_at).total_seconds() / 86400.0)


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def detect_red_flags(
    context: CandidateContext,
    scores: ScoreComponents,
    *,
    now: datetime | None = None,
) -> list[RedFlag]:
    """Compute deterministic red flags from evidence and the score breakdown.

    Runs before any LLM call. Detection lives in code, not in a prompt, so a
    known disqualifier can never be missed because a model overlooked it.
    """

    as_of = now or context.as_of
    if as_of.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    evidence = context.evidence
    flags: list[RedFlag] = []

    # --- dilution overhang: shelf / ATM / offering / warrant ---
    dilution = [
        item
        for item in evidence
        if item.source_type in DILUTION_TYPES
        or _contains(item.observed_fact, DILUTION_KEYWORDS)
    ]
    if dilution:
        fresh = any(_age_days(item, as_of) <= DILUTION_FRESH_DAYS for item in dilution)
        flags.append(
            RedFlag(
                code="dilution_overhang",
                severity="critical" if fresh else "warn",
                message=(
                    f"{len(dilution)} dilution-related item(s) (shelf/ATM/offering/"
                    "warrant). New issuance can cap a long call's upside."
                ),
                evidence_ids=[item.evidence_id for item in dilution],
            )
        )

    # --- insider selling reported on Form 4 ---
    insider = [
        item
        for item in evidence
        if item.source_type == "sec_form4"
        and _contains(item.observed_fact, INSIDER_SELL_KEYWORDS)
    ]
    if insider:
        fresh = any(_age_days(item, as_of) <= INSIDER_SELL_FRESH_DAYS for item in insider)
        flags.append(
            RedFlag(
                code="insider_selling",
                severity="critical" if fresh else "warn",
                message="Insider sale(s) reported on Form 4 alongside the bull thesis.",
                evidence_ids=[item.evidence_id for item in insider],
            )
        )

    # --- earnings inside the window: IV crush can sink a long option ---
    earnings = [item for item in evidence if item.source_type == "earnings_calendar"]
    if earnings:
        flags.append(
            RedFlag(
                code="earnings_iv_crush",
                severity="warn",
                message=(
                    "Earnings event present: a long option can lose to IV crush "
                    "even when the direction is right."
                ),
                evidence_ids=[item.evidence_id for item in earnings],
            )
        )

    # --- stale catalyst: the freshest evidence is already old ---
    if evidence:
        freshest = min(_age_days(item, as_of) for item in evidence)
        if freshest > STALE_CATALYST_DAYS:
            flags.append(
                RedFlag(
                    code="stale_catalyst",
                    severity="warn",
                    message=(
                        f"Freshest evidence is {freshest:.0f} days old "
                        f"(> {STALE_CATALYST_DAYS:.0f}); the catalyst may be priced in."
                    ),
                )
            )

    # --- thin evidence: a single item is low corroboration ---
    if len(evidence) < 2:
        flags.append(
            RedFlag(
                code="thin_evidence",
                severity="warn",
                message="Thesis rests on a single evidence item; low corroboration.",
                evidence_ids=[item.evidence_id for item in evidence],
            )
        )

    # --- no primary source: only news / flow, no filing or calendar ---
    if evidence and not any(item.source_type in _PRIMARY_SOURCES for item in evidence):
        flags.append(
            RedFlag(
                code="no_primary_source",
                severity="warn",
                message=(
                    "No SEC filing / IR / calendar corroboration; thesis rests on "
                    "news or options flow only."
                ),
            )
        )

    # --- weak deterministic catalyst score ---
    if scores.catalyst < WEAK_CATALYST_SCORE:
        flags.append(
            RedFlag(
                code="weak_catalyst_score",
                severity="warn",
                message=f"Deterministic catalyst score is low ({scores.catalyst}).",
            )
        )

    return flags


def critical_flags(flags: list[RedFlag]) -> list[RedFlag]:
    return [flag for flag in flags if flag.severity == "critical"]


def format_red_flags(flags: list[RedFlag]) -> str:
    """Render flags as a briefing block for the skeptic / risk_manager roles."""

    if not flags:
        return "Deterministic red flags: none detected by code."
    lines = [
        "Deterministic red flags already detected by code (treat as observed"
        " facts, not opinion):"
    ]
    for flag in flags:
        cited = f" [evidence: {', '.join(flag.evidence_ids)}]" if flag.evidence_ids else ""
        lines.append(f"  - ({flag.severity.upper()}) {flag.code}: {flag.message}{cited}")
    return "\n".join(lines)
