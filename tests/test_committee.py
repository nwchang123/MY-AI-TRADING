import json
from datetime import datetime, timezone

from trading_agent.domain.evidence import CandidateContext, EvidenceItem
from trading_agent.research.committee import Committee
from trading_agent.research.llm import MockLLMClient
from trading_agent.research.scoring import ScoreInputs, score_candidate

NOW = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)

_PROPOSAL = {
    "decision": "open_position",
    "ticker": "EXAMPLE",
    "option_code": "US.EXAMPLE260626C00005000",
    "option_side": "call",
    "action": "buy_to_open",
    "contracts": 1,
    "limit_price": 0.2,
    "max_limit_price": 0.21,
    "thesis": "Named supply agreement catalyst with bounded premium.",
    "evidence_ids": ["evt_1"],
    "confidence": 0.7,
    "expected_catalyst_window": "2026-06-10/2026-06-20",
    "exit_plan": {"take_profit_pct": 100, "stop_loss_pct": 50, "time_stop": "2026-06-24"},
    "invalidation": ["Catalyst delayed"],
}


def _context() -> CandidateContext:
    return CandidateContext(
        ticker="EXAMPLE",
        as_of=NOW,
        evidence=[
            EvidenceItem(
                evidence_id="evt_1",
                ticker="EXAMPLE",
                source_type="sec_8k",
                source_url="https://sec.gov/x",
                published_at=NOW,
                observed_fact="Named multi-year supply agreement filed in an 8-K.",
                retrieved_at=NOW,
            )
        ],
    )


def _scores():
    return score_candidate(
        ScoreInputs(catalyst=0.8, options=0.6, underlying=0.5, operations=0.6, contradictions=0.1)
    )


def _run(responses: list[str]):
    client = MockLLMClient(responses)
    return Committee(client).run(_context(), _scores())


def test_open_position_flows_through() -> None:
    out = _run(
        ["catalyst note", "options note", "looks fine", "acceptable", json.dumps(_PROPOSAL)]
    )
    assert out.decision == "open_position"
    assert out.proposal is not None
    assert out.proposal.option_code == "US.EXAMPLE260626C00005000"
    assert out.llm_calls == 5
    assert out.vetoes == []


def test_skeptic_veto_blocks_open() -> None:
    out = _run(
        ["catalyst", "options", "VETO: dilution shelf detected", "ok", json.dumps(_PROPOSAL)]
    )
    assert out.decision == "reject"
    assert out.proposal is None
    assert any("skeptic" in v for v in out.vetoes)


def test_risk_manager_veto_blocks_open() -> None:
    out = _run(
        ["catalyst", "options", "fine", "VETO: downside unbounded", json.dumps(_PROPOSAL)]
    )
    assert out.decision == "reject"
    assert any("risk_manager" in v for v in out.vetoes)


def test_unknown_evidence_is_rejected() -> None:
    bad = dict(_PROPOSAL, evidence_ids=["evt_999"])
    out = _run(["c", "o", "fine", "fine", json.dumps(bad)])
    assert out.decision == "reject"
    assert "unknown evidence" in out.rationale


def test_ticker_mismatch_is_rejected() -> None:
    bad = dict(_PROPOSAL, ticker="OTHER")
    out = _run(["c", "o", "fine", "fine", json.dumps(bad)])
    assert out.decision == "reject"
    assert "does not match" in out.rationale


def test_invalid_json_is_rejected_not_raised() -> None:
    out = _run(["c", "o", "fine", "fine", "this is not json at all"])
    assert out.decision == "reject"
    assert "invalid JSON" in out.rationale


def test_schema_violation_is_rejected() -> None:
    bad = dict(_PROPOSAL, contracts=5)  # mandate is one contract; schema requires >0 but gate caps
    bad["confidence"] = 2.0  # out of range -> schema rejects
    out = _run(["c", "o", "fine", "fine", json.dumps(bad)])
    assert out.decision == "reject"
    assert "schema validation" in out.rationale


def test_hold_decision_passes_through() -> None:
    out = _run(["c", "o", "fine", "fine", json.dumps({"decision": "hold", "rationale": "stale"})])
    assert out.decision == "hold"
    assert out.proposal is None
    assert out.rationale == "stale"


def test_json_code_fences_are_stripped() -> None:
    fenced = "```json\n" + json.dumps(_PROPOSAL) + "\n```"
    out = _run(["c", "o", "fine", "fine", fenced])
    assert out.decision == "open_position"
    assert out.proposal is not None
