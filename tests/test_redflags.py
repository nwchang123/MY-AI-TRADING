from datetime import datetime, timedelta, timezone

from trading_agent.domain.evidence import CandidateContext, EvidenceItem, SourceType
from trading_agent.research.redflags import (
    bearish_critical_flags,
    critical_flags,
    detect_red_flags,
    format_red_flags,
    nondirectional_critical_flags,
)
from trading_agent.research.scoring import ScoreInputs, score_candidate

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)


def _evidence(
    evidence_id: str,
    source_type: SourceType,
    fact: str,
    *,
    age_days: float = 0.0,
) -> EvidenceItem:
    published = NOW - timedelta(days=age_days)
    return EvidenceItem(
        evidence_id=evidence_id,
        ticker="EXAMPLE",
        source_type=source_type,
        source_url=f"https://sec.gov/{evidence_id}",
        published_at=published,
        observed_fact=fact,
        retrieved_at=NOW,
    )


def _context(items: list[EvidenceItem]) -> CandidateContext:
    return CandidateContext(ticker="EXAMPLE", as_of=NOW, evidence=items)


def _scores(catalyst: float = 0.8):
    return score_candidate(
        ScoreInputs(catalyst=catalyst, operations=0.6, contradictions=0.1)
    )


def _codes(flags) -> set[str]:
    return {flag.code for flag in flags}


def test_fresh_shelf_is_warn_not_critical() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_s3", "Filed an S-3 shelf registration.", age_days=3),
            _evidence("e2", "sec_8k", "Named supply agreement.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    dilution = next(f for f in flags if f.code == "dilution_overhang")
    assert dilution.severity == "warn"
    assert dilution.evidence_ids == ["e1"]
    assert not critical_flags(flags)


def test_dilution_keyword_in_8k_is_detected() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_8k", "Announced an at-the-market offering.", age_days=2),
            _evidence("e2", "sec_8k", "Other news.", age_days=2),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert "dilution_overhang" in _codes(flags)


def test_old_dilution_downgrades_to_warning() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_s3", "Filed an S-3 shelf registration.", age_days=200),
            _evidence("e2", "sec_8k", "Named supply agreement.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    dilution = next(f for f in flags if f.code == "dilution_overhang")
    assert dilution.severity == "warn"
    assert not critical_flags(flags)


def test_fresh_insider_selling_is_critical() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_form4", "Officer sold 50,000 shares.", age_days=5),
            _evidence("e2", "sec_8k", "Named supply agreement.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    insider = next(f for f in flags if f.code == "insider_selling")
    assert insider.severity == "critical"


def test_form4_without_sell_keyword_is_not_flagged() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_form4", "Officer acquired 50,000 shares.", age_days=5),
            _evidence("e2", "sec_8k", "Named supply agreement.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert "insider_selling" not in _codes(flags)


def test_earnings_in_window_warns_about_iv_crush() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_8k", "Named supply agreement.", age_days=1),
            _evidence("e2", "earnings_calendar", "Earnings on 2026-06-18.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    iv = next(f for f in flags if f.code == "earnings_iv_crush")
    assert iv.severity == "warn"
    assert not critical_flags(flags)


def test_earnings_8k_item_202_fires_iv_crush_without_calendar() -> None:
    # An 8-K item 2.02 IS the earnings release; the labeller stamps "(earnings)"
    # so the IV-crush flag fires with no separate earnings-calendar feed.
    ctx = _context(
        [
            _evidence(
                "e1", "sec_8k",
                "SEC 8-K filing; items: 2.02 Results of Operations (earnings) "
                "[signals: earnings].",
                age_days=1,
            ),
            _evidence("e2", "press_release", "Other news.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert "earnings_iv_crush" in _codes(flags)


def test_bearish_and_nondirectional_flag_partition() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_424b", "Filed an offering prospectus.", age_days=3),
            _evidence("e2", "sec_form4", "Officer sold 50,000 shares.", age_days=5),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    bearish = {f.code for f in bearish_critical_flags(flags)}
    assert bearish == {"dilution_overhang", "insider_selling"}
    # Both criticals are directional, so nothing hard-blocks regardless.
    assert nondirectional_critical_flags(flags) == []


def test_thin_single_item_evidence_warns() -> None:
    ctx = _context([_evidence("e1", "sec_8k", "Named supply agreement.", age_days=1)])
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert "thin_evidence" in _codes(flags)


def test_no_primary_source_warns() -> None:
    ctx = _context(
        [
            _evidence("e1", "moomoo_news", "Blog says a deal is coming.", age_days=1),
            _evidence("e2", "options_flow", "Unusual call volume.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert "no_primary_source" in _codes(flags)


def test_stale_catalyst_warns() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_8k", "Named supply agreement.", age_days=60),
            _evidence("e2", "press_release", "Old news.", age_days=90),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert "stale_catalyst" in _codes(flags)


def test_weak_catalyst_score_warns() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_8k", "Named supply agreement.", age_days=1),
            _evidence("e2", "press_release", "More news.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(catalyst=0.1), now=NOW)
    assert "weak_catalyst_score" in _codes(flags)


def test_clean_candidate_has_no_critical_flags() -> None:
    ctx = _context(
        [
            _evidence("e1", "sec_8k", "Named multi-year supply agreement.", age_days=1),
            _evidence("e2", "press_release", "Confirms revenue next quarter.", age_days=1),
        ]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    assert not critical_flags(flags)


def test_format_red_flags_lists_severity_and_codes() -> None:
    ctx = _context(
        [_evidence("e1", "sec_424b", "Filed an offering prospectus.", age_days=3)]
    )
    flags = detect_red_flags(ctx, _scores(), now=NOW)
    rendered = format_red_flags(flags)
    assert "CRITICAL" in rendered
    assert "dilution_overhang" in rendered


def test_format_red_flags_handles_empty() -> None:
    assert "none detected" in format_red_flags([])


def test_naive_now_is_rejected() -> None:
    ctx = _context([_evidence("e1", "sec_8k", "News.", age_days=1)])
    try:
        detect_red_flags(ctx, _scores(), now=datetime(2026, 6, 2, 15, 0))
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for naive datetime")
