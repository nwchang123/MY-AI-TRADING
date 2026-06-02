from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.domain.evidence import EvidenceItem, SourceType
from trading_agent.research.catalysts import (
    build_candidate_context,
    dedupe_evidence,
    derive_score_inputs,
)
from trading_agent.research.scoring import score_candidate

AS_OF = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)


def _ev(
    evidence_id: str,
    source_type: SourceType,
    *,
    fact: str = "SEC 8-K filing accepted by EDGAR.",
    url: str | None = None,
    age_days: float = 1.0,
) -> EvidenceItem:
    published = AS_OF - timedelta(days=age_days)
    return EvidenceItem(
        evidence_id=evidence_id,
        ticker="EXAMPLE",
        source_type=source_type,
        source_url=url or f"https://sec.gov/{evidence_id}",
        published_at=published,
        observed_fact=fact,
        retrieved_at=AS_OF,
    )


def test_dedupe_collapses_repeated_urls() -> None:
    a = _ev("e1", "press_release", fact="Story one.", url="https://x.com/story")
    b = _ev("e2", "press_release", fact="Story one.", url="https://x.com/story")  # syndicated dup
    c = _ev("e3", "press_release", fact="Different story.", url="https://x.com/other")
    assert len(dedupe_evidence([a, b, c])) == 2


def test_dedupe_collapses_identical_fact_across_urls() -> None:
    a = _ev("e1", "press_release", fact="Same headline.", url="https://a.com/1")
    b = _ev("e2", "press_release", fact="Same headline.", url="https://b.com/2")
    assert len(dedupe_evidence([a, b])) == 1


def test_fresh_catalyst_scores_higher_than_stale() -> None:
    fresh = derive_score_inputs([_ev("e1", "sec_8k", age_days=1)], AS_OF)
    stale = derive_score_inputs([_ev("e2", "sec_8k", age_days=29)], AS_OF)
    assert fresh.catalyst > stale.catalyst


def test_stale_evidence_does_not_inflate() -> None:
    # Older than the catalyst horizon -> zero contribution.
    stale = derive_score_inputs([_ev("e1", "sec_8k", age_days=60)], AS_OF)
    assert stale.catalyst == 0.0


def test_partnership_keyword_boosts_catalyst() -> None:
    plain = derive_score_inputs([_ev("e1", "sec_8k", fact="Routine 8-K.", age_days=2)], AS_OF)
    partner = derive_score_inputs(
        [_ev("e2", "sec_8k", fact="Named multi-year supply agreement signed.", age_days=2)],
        AS_OF,
    )
    assert partner.catalyst > plain.catalyst
    assert partner.operations > 0  # supply agreement is also an operations signal


def test_dilution_filing_reduces_total_score() -> None:
    catalyst_only = [_ev("e1", "sec_8k", fact="Named supply agreement.", age_days=2)]
    with_dilution = catalyst_only + [_ev("e2", "sec_s3", fact="Shelf registration.", age_days=2)]

    clean = score_candidate(derive_score_inputs(catalyst_only, AS_OF))
    dirty = score_candidate(derive_score_inputs(with_dilution, AS_OF))

    assert dirty.contradictions < 0  # dilution penalty applied
    assert dirty.total < clean.total


def test_options_and_underlying_pass_through() -> None:
    inputs = derive_score_inputs(
        [_ev("e1", "sec_8k", age_days=1)], AS_OF, options=0.6, underlying=0.4
    )
    assert inputs.options == 0.6
    assert inputs.underlying == 0.4


def test_build_candidate_context_requires_evidence() -> None:
    with pytest.raises(ValueError, match="no usable evidence"):
        build_candidate_context("EXAMPLE", [], AS_OF)


def test_build_candidate_context_dedupes() -> None:
    a = _ev("e1", "sec_8k", url="https://sec.gov/same")
    b = _ev("e2", "sec_8k", url="https://sec.gov/same")
    context = build_candidate_context("example", [a, b], AS_OF)
    assert context.ticker == "EXAMPLE"
    assert len(context.evidence) == 1
