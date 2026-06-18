from __future__ import annotations

from datetime import datetime

from trading_agent.domain.evidence import CandidateContext, EvidenceItem, SourceType
from trading_agent.research.scoring import ScoreInputs

# Fresh-catalyst window and the longer window over which a dilution overhang
# still matters. Older evidence decays toward zero so stale stories and
# duplicates cannot inflate a score.
CATALYST_HORIZON_DAYS = 30.0
CONTRADICTION_HORIZON_DAYS = 120.0

# Per-type catalyst strength. Routine periodic reports count for little; 8-Ks,
# press releases, and regulatory dates carry the real catalysts.
_CATALYST_TYPE_WEIGHT: dict[SourceType, float] = {
    "sec_8k": 1.0,
    "press_release": 1.0,
    "regulatory_calendar": 1.0,
    "investor_relations": 0.9,
    "earnings_calendar": 0.9,
    "sec_13d": 0.9,
    "moomoo_news": 0.8,
    # Aggregated headlines (Google News etc.): real catalysts surface here first,
    # but syndicated noise does too, so they weigh below primary sources.
    "news_rss": 0.7,
    "options_flow": 0.8,
    "sec_10q": 0.5,
    "sec_10k": 0.4,
}

# Public so the deterministic red-flag detector reuses the exact same
# dilution / insider-selling vocabulary as the scorer (single source of truth).
DILUTION_TYPES: set[SourceType] = {"sec_s3", "sec_424b"}

DILUTION_KEYWORDS = (
    "dilution",
    "at-the-market",
    "atm offering",
    "offering",
    "registered direct",
    "warrant",
    "public offering",
    "private placement",
)
INSIDER_SELL_KEYWORDS = ("sold", "sale of", "disposed", "insider selling")
_OPERATIONS_KEYWORDS = (
    "hiring",
    "hire",
    "capex",
    "capacity",
    "facility",
    "procurement",
    "partnership",
    "partner",
    "supply agreement",
    "purchase order",
    "expansion",
    "contract award",
)
_PARTNERSHIP_BOOST_KEYWORDS = (
    "supply agreement",
    "partnership",
    "named customer",
    "contract award",
    "approval",
    "clearance",
    "fda",
)


def dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Drop duplicate source URLs and identical (type, fact) pairs.

    Keeps the first occurrence so repeated syndication of one story counts once.
    """

    seen_urls: set[str] = set()
    seen_facts: set[tuple[str, str]] = set()
    deduped: list[EvidenceItem] = []
    for item in items:
        fact_key = (item.source_type, item.observed_fact.strip().lower())
        if item.source_url in seen_urls or fact_key in seen_facts:
            continue
        seen_urls.add(item.source_url)
        seen_facts.add(fact_key)
        deduped.append(item)
    return deduped


def _age_days(item: EvidenceItem, as_of: datetime) -> float:
    return max(0.0, (as_of - item.published_at).total_seconds() / 86400.0)


def _recency_weight(item: EvidenceItem, as_of: datetime, horizon_days: float) -> float:
    return max(0.0, 1.0 - _age_days(item, as_of) / horizon_days)


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def derive_score_inputs(items: list[EvidenceItem], as_of: datetime) -> ScoreInputs:
    """Deterministically map catalyst evidence to score sub-inputs.

    Fills ``catalyst``, ``operations``, and ``contradictions`` from the public
    evidence. (Live option-chain / underlying market-data sub-scores are not
    derived here and are not part of the score until that feed is wired.)
    """

    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    items = dedupe_evidence(items)

    catalyst_scores: list[float] = []
    operations_scores: list[float] = []
    contradiction_scores: list[float] = []
    dilution_count = 0

    for item in items:
        catalyst_w = _recency_weight(item, as_of, CATALYST_HORIZON_DAYS)
        contradiction_w = _recency_weight(item, as_of, CONTRADICTION_HORIZON_DAYS)

        type_weight = _CATALYST_TYPE_WEIGHT.get(item.source_type)
        if type_weight is not None:
            base = catalyst_w * type_weight
            if catalyst_w > 0.3 and _contains(item.observed_fact, _PARTNERSHIP_BOOST_KEYWORDS):
                base = min(1.0, base + 0.2)
            catalyst_scores.append(base)

        if _contains(item.observed_fact, _OPERATIONS_KEYWORDS):
            operations_scores.append(catalyst_w)

        is_dilution = item.source_type in DILUTION_TYPES or _contains(
            item.observed_fact, DILUTION_KEYWORDS
        )
        is_insider_sale = item.source_type == "sec_form4" and _contains(
            item.observed_fact, INSIDER_SELL_KEYWORDS
        )
        if is_dilution:
            dilution_count += 1
        if is_dilution or is_insider_sale:
            contradiction_scores.append(contradiction_w)

    catalyst = min(1.0, max(catalyst_scores, default=0.0))
    operations = min(1.0, max(operations_scores, default=0.0))
    contradictions = max(contradiction_scores, default=0.0)
    if dilution_count >= 2:
        contradictions += 0.25
    contradictions = min(1.0, contradictions)

    return ScoreInputs(
        catalyst=round(catalyst, 4),
        operations=round(operations, 4),
        contradictions=round(contradictions, 4),
    )


def build_candidate_context(
    ticker: str, items: list[EvidenceItem], as_of: datetime
) -> CandidateContext:
    deduped = dedupe_evidence(items)
    if not deduped:
        raise ValueError(f"no usable evidence collected for {ticker}")
    return CandidateContext(ticker=ticker.upper(), as_of=as_of, evidence=deduped)
