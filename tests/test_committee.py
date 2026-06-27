import json
from datetime import date, datetime, timezone

from trading_agent.domain.evidence import CandidateContext, EvidenceItem
from trading_agent.domain.proposals import ExitPlan, OptionCandidate
from trading_agent.research.committee import Committee, parse_win_prob
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
    return score_candidate(ScoreInputs(catalyst=0.8, operations=0.6, contradictions=0.1))


def _run(responses: list[str]):
    client = MockLLMClient(responses)
    return Committee(client).run(_context(), _scores())


def test_exit_plan_normalizes_signed_fractional_stop_loss():
    """The committee sometimes expresses the stop as a signed fraction (-0.5 =
    '-50%'); the schema wants a positive magnitude. Coerce instead of failing
    validation and killing an otherwise-valid proposal (last night: 6 lost)."""
    ep = ExitPlan(take_profit_pct=1.0, stop_loss_pct=-0.5, time_stop=date(2026, 7, 17))
    assert ep.take_profit_pct == 100.0
    assert ep.stop_loss_pct == 50.0
    # A plain magnitude passes through unchanged.
    ep2 = ExitPlan(take_profit_pct=100.0, stop_loss_pct=50.0, time_stop=date(2026, 7, 17))
    assert ep2.stop_loss_pct == 50.0


def test_iv_ramp_briefing_states_confirmed_earnings_date():
    """A known earnings date is handed to the committee as fact so no role
    re-derives (and rejects on) a guessed date -- last night's dominant veto."""
    c = Committee(
        MockLLMClient([]),
        pre_earnings_exit_trading_days=2,
        earnings_window_max_days=25,
    )
    block = c._format_strategy(date(2026, 7, 24))
    assert "2026-07-24" in block
    assert "treat it as FACT" in block
    assert "expires AFTER this date" in block


def test_iv_ramp_briefing_tolerates_missing_earnings_date():
    """A momentum-backfill name without a confirmed date must NOT be vetoed
    solely for lacking one."""
    c = Committee(
        MockLLMClient([]),
        pre_earnings_exit_trading_days=2,
        earnings_window_max_days=25,
    )
    block = c._format_strategy(None)
    assert "could not confirm an exact earnings date" in block
    assert "Do NOT veto SOLELY" in block


def _candidates() -> list[OptionCandidate]:
    return [
        OptionCandidate(
            option_code="US.EXAMPLE260626C00005000",
            option_side="call",
            strike=5.0,
            expiry=date(2026, 6, 26),
            bid=0.19,
            ask=0.21,
            open_interest=200,
            daily_volume=30,
            iv=0.5,
            dte=24,
            estimated_contract_cost_usd=22.0,
        )
    ]


def _run_with_candidates(responses: list[str], candidates: list[OptionCandidate]):
    client = MockLLMClient(responses)
    output = Committee(client).run(_context(), _scores(), candidates=candidates)
    return client, output


def test_open_position_flows_through() -> None:
    out = _run(
        ["catalyst note", "options note", "looks fine", "acceptable", json.dumps(_PROPOSAL)]
    )
    assert out.decision == "open_position"
    assert out.proposal is not None
    assert out.proposal.option_code == "US.EXAMPLE260626C00005000"
    assert out.llm_calls == 5
    assert out.vetoes == []


def test_iv_ramp_strategy_note_briefed_when_entry_window_active() -> None:
    # IV-ramp ENTRY active (earnings_window_max_days > 0) -> full IV-ramp framing.
    client = MockLLMClient(["c", "o", "s", "r", json.dumps(_PROPOSAL)])
    Committee(
        client, pre_earnings_exit_trading_days=2, earnings_window_max_days=25
    ).run(_context(), _scores())
    briefing = client.calls[0]["user"]  # the catalyst analyst's briefing
    assert "PRE-EARNINGS IV-RAMP" in briefing
    assert "do NOT veto on event/earnings IV-crush" in briefing


def test_catalyst_note_when_only_exit_guard_kept() -> None:
    # Regression: exit guard ON but IV-ramp ENTRY OFF (window == 0). The briefing
    # must NOT frame the trade as a pre-earnings IV-ramp entry (that vetoed every
    # volume-ratio name for lacking an upcoming earnings date); it must say it is
    # a catalyst play with no earnings-date / post-earnings-expiry requirement.
    client = MockLLMClient(["c", "o", "s", "r", json.dumps(_PROPOSAL)])
    Committee(
        client, pre_earnings_exit_trading_days=2, earnings_window_max_days=0
    ).run(_context(), _scores())
    briefing = client.calls[0]["user"]
    assert "PRE-EARNINGS IV-RAMP" not in briefing
    assert "NOT an earnings play" in briefing
    assert "NO requirement for an upcoming earnings date" in briefing


def test_no_strategy_note_when_all_disabled() -> None:
    client = MockLLMClient(["c", "o", "s", "r", json.dumps(_PROPOSAL)])
    Committee(client).run(_context(), _scores())  # default: everything off
    assert "PRE-EARNINGS IV-RAMP" not in client.calls[0]["user"]
    assert "NOT an earnings play" not in client.calls[0]["user"]


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


def test_parse_win_prob_variants() -> None:
    assert parse_win_prob("analysis...\nWIN_PROB: 0.55") == 0.55
    assert parse_win_prob("win_prob: 55%") == 0.55
    assert parse_win_prob("WIN_PROB: 70") == 0.70  # bare number above 1 -> percent
    assert parse_win_prob("WIN_PROB：0.4") == 0.4  # full-width colon
    assert parse_win_prob("first WIN_PROB: 0.3 then WIN_PROB: 0.6") == 0.6  # last wins
    assert parse_win_prob("no estimate here") is None
    assert parse_win_prob("") is None


def _run_with_threshold(responses: list[str], threshold: float):
    client = MockLLMClient(responses)
    committee = Committee(client, min_win_probability=threshold)
    return committee.run(_context(), _scores())


def test_win_probability_floor_passes_confident_trade() -> None:
    out = _run_with_threshold(
        [
            "catalyst",
            "options",
            "looks fine\nWIN_PROB: 0.65",
            "acceptable\nWIN_PROB: 0.60",
            json.dumps(_PROPOSAL),  # PM confidence 0.7
        ],
        threshold=0.55,
    )
    assert out.decision == "open_position"
    assert out.win_probability == 0.60  # the lowest estimate binds
    assert out.win_estimates["risk_manager"] == 0.60
    assert out.win_estimates["portfolio_manager"] == 0.7


def test_one_pessimistic_lineage_blocks_the_trade() -> None:
    out = _run_with_threshold(
        [
            "catalyst",
            "options",
            "weak setup\nWIN_PROB: 0.30",  # skeptic (adversary lineage) says no
            "acceptable\nWIN_PROB: 0.70",
            json.dumps(_PROPOSAL),
        ],
        threshold=0.55,
    )
    assert out.decision == "reject"
    assert "0.30" in out.rationale and "0.55" in out.rationale
    assert out.win_probability == 0.30


def test_missing_win_prob_counts_as_zero() -> None:
    out = _run_with_threshold(
        [
            "catalyst",
            "options",
            "looks fine",  # forgot WIN_PROB -> treated as 0
            "acceptable\nWIN_PROB: 0.70",
            json.dumps(_PROPOSAL),
        ],
        threshold=0.55,
    )
    assert out.decision == "reject"
    assert out.win_probability == 0.0
    assert out.win_estimates["skeptic"] is None


def test_zero_threshold_disables_the_check() -> None:
    out = _run_with_threshold(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)],
        threshold=0.0,
    )
    assert out.decision == "open_position"


def test_candidate_mode_accepts_listed_code() -> None:
    _client, out = _run_with_candidates(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)],
        _candidates(),
    )
    assert out.decision == "open_position"
    assert out.proposal is not None
    assert out.proposal.option_code == "US.EXAMPLE260626C00005000"


def test_candidate_mode_rejects_code_not_in_list() -> None:
    # PM names a contract that is not among the supplied candidates -> reject.
    bad = dict(_PROPOSAL, option_code="US.EXAMPLE260626C00009000")
    _client, out = _run_with_candidates(
        ["catalyst", "options", "fine", "fine", json.dumps(bad)],
        _candidates(),
    )
    assert out.decision == "reject"
    assert "candidate" in out.rationale.lower()


def test_market_snapshot_appears_in_briefing() -> None:
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    Committee(client).run(
        _context(),
        _scores(),
        market_snapshot={
            "price": 18.4,
            "change_pct": 6.3,
            "prev_close": 17.3,
            "day_high": 18.9,
            "day_low": 17.1,
            "volume": 2000000,
            "iv30": 0.66,
            "iv30_change": 0.04,
        },
    )
    briefing = client.calls[0]["user"]
    assert "Underlying market snapshot" in briefing
    assert "price=18.4" in briefing
    assert "day_change=+6.30%" in briefing


def test_price_context_appears_in_briefing() -> None:
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    Committee(client).run(
        _context(),
        _scores(),
        price_context={
            "last_close": 18.4,
            "ret_5d": 42.0,
            "ret_20d": 310.0,
            "realized_vol_20d": 95.0,
            "pct_from_20d_high": -2.0,
            "pct_from_20d_low": 280.0,
        },
    )
    briefing = client.calls[0]["user"]
    assert "technical context" in briefing
    assert "ret_20d=310.0%" in briefing
    assert "realized_vol_20d=95.0%" in briefing


def test_committee_rules_warn_against_evidence_injection() -> None:
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    Committee(client).run(_context(), _scores())
    system = client.calls[0]["system"]  # catalyst analyst shares the common rules
    assert "untrusted DATA" in system
    assert "manipulation red flag" in system


def test_empty_market_snapshot_leaves_briefing_unchanged() -> None:
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    Committee(client).run(_context(), _scores(), market_snapshot={})
    assert "Underlying market snapshot" not in client.calls[0]["user"]


def test_candidate_list_is_shown_to_options_and_pm() -> None:
    client, _out = _run_with_candidates(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)],
        _candidates(),
    )
    options_prompt = client.calls[1]["user"]
    pm_prompt = client.calls[4]["user"]
    assert "CANDIDATE CONTRACTS" in options_prompt
    assert "US.EXAMPLE260626C00005000" in options_prompt
    assert "US.EXAMPLE260626C00005000" in pm_prompt


def _dilution_context() -> CandidateContext:
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
            ),
            EvidenceItem(
                evidence_id="evt_2",
                ticker="EXAMPLE",
                source_type="sec_424b",
                source_url="https://sec.gov/424b",
                published_at=NOW,
                observed_fact="Filed an offering prospectus for up to USD 100M.",
                retrieved_at=NOW,
            ),
        ],
    )


def test_critical_red_flag_short_circuits_without_llm_calls() -> None:
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    out = Committee(client).run(_dilution_context(), _scores())
    assert out.decision == "reject"
    assert out.proposal is None
    assert out.llm_calls == 0
    assert client.calls == []  # committee skipped entirely
    assert any("redflags" in v for v in out.vetoes)
    assert any(f.code == "dilution_overhang" for f in out.red_flags)


_PUT_PROPOSAL = dict(
    _PROPOSAL,
    option_code="US.EXAMPLE260626P00005000",
    option_side="put",
    thesis="Fresh dilution offering caps upside; long put on the overhang.",
)


def _put_candidates() -> list[OptionCandidate]:
    return [
        OptionCandidate(
            option_code="US.EXAMPLE260626P00005000",
            option_side="put",
            strike=5.0,
            expiry=date(2026, 6, 26),
            bid=0.19,
            ask=0.21,
            open_interest=200,
            daily_volume=30,
            iv=0.5,
            dte=24,
            estimated_contract_cost_usd=22.0,
        )
    ]


def test_bearish_flag_runs_puts_only_and_accepts_a_put() -> None:
    # A fresh dilution offering no longer kills the name: with a put on the chain
    # the committee runs and a long-put thesis flows through.
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PUT_PROPOSAL)]
    )
    out = Committee(client).run(
        _dilution_context(), _scores(), candidates=_put_candidates()
    )
    assert out.decision == "open_position"
    assert out.proposal is not None and out.proposal.option_side == "put"
    assert out.llm_calls == 5  # committee actually ran, not free-rejected
    assert "DIRECTIONAL CONSTRAINT" in client.calls[1]["user"]  # options analyst saw it


def test_bearish_flag_blocks_a_call_even_with_put_candidates() -> None:
    # Both sides on the chain, but a bearish critical filters the list to puts;
    # a call proposal cannot get through.
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    out = Committee(client).run(
        _dilution_context(), _scores(), candidates=_put_candidates() + _candidates()
    )
    assert out.decision == "reject"
    assert out.proposal is None


def test_bearish_flag_with_no_put_candidate_hard_blocks_without_llm() -> None:
    # Only a call is tradeable: nothing to express the downside -> zero-API block.
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    out = Committee(client).run(_dilution_context(), _scores(), candidates=_candidates())
    assert out.decision == "reject"
    assert out.llm_calls == 0
    assert client.calls == []


def test_warn_red_flags_do_not_block_open() -> None:
    # The single-item _context() yields only a thin_evidence warning.
    out = _run(
        ["catalyst note", "options note", "looks fine", "acceptable", json.dumps(_PROPOSAL)]
    )
    assert out.decision == "open_position"
    assert out.llm_calls == 5
    assert any(f.code == "thin_evidence" for f in out.red_flags)
    assert all(f.severity == "warn" for f in out.red_flags)


def test_warn_red_flags_are_passed_to_the_skeptic() -> None:
    client = MockLLMClient(
        ["catalyst", "options", "fine", "fine", json.dumps(_PROPOSAL)]
    )
    Committee(client).run(_context(), _scores())
    skeptic_prompt = client.calls[2]["user"]
    assert "Deterministic red flags" in skeptic_prompt
    assert "thin_evidence" in skeptic_prompt


def _named(responses: list[str], model: str) -> MockLLMClient:
    client = MockLLMClient(responses)
    client.model = model  # exercised by RoleNote model attribution
    return client


def test_adversary_client_runs_skeptic_and_risk() -> None:
    primary = _named(["catalyst note", "options note"], "primary-flash")
    adversary = _named(["skeptic ok", "risk ok"], "adversary-model")
    pro = _named([json.dumps(_PROPOSAL)], "pro-model")

    out = Committee(primary, pro_client=pro, adversary_client=adversary).run(
        _context(), _scores()
    )

    assert out.decision == "open_position"
    # Idea roles hit the primary, veto roles hit the adversary, PM hits pro.
    assert len(primary.calls) == 2
    assert len(adversary.calls) == 2
    assert len(pro.calls) == 1
    roles = {note.role: note.model for note in out.role_notes}
    assert roles["skeptic"] == "adversary-model"
    assert roles["risk_manager"] == "adversary-model"
    assert roles["catalyst_analyst"] == "primary-flash"
    assert roles["portfolio_manager"] == "pro-model"


def test_adversary_veto_blocks_open() -> None:
    primary = MockLLMClient(["catalyst", "options"])
    adversary = MockLLMClient(["VETO: dilution shelf detected", "ok"])
    pro = MockLLMClient([json.dumps(_PROPOSAL)])

    out = Committee(primary, pro_client=pro, adversary_client=adversary).run(
        _context(), _scores()
    )
    assert out.decision == "reject"
    assert any("skeptic" in v for v in out.vetoes)


def _proposal_with_confidence(conf: float) -> str:
    payload = dict(_PROPOSAL)
    payload["confidence"] = conf
    return json.dumps(payload)


def test_soft_veto_lets_pm_override_with_high_conviction() -> None:
    # penalty 0.10 + floor 0.55: one veto needs the non-vetoing roles + PM to
    # clear 0.65. The skeptic vetoes (its 0.15 drops out, becomes a penalty);
    # risk + PM are confident, so binding = min(0.72, 0.72) - 0.10 = 0.62 >= 0.55.
    client = MockLLMClient(
        [
            "catalyst",
            "options",
            "VETO: dilution risk\nWIN_PROB: 0.15",  # skeptic vetoes
            "acceptable\nWIN_PROB: 0.72",  # risk does NOT veto
            _proposal_with_confidence(0.72),
        ]
    )
    out = Committee(
        client, min_win_probability=0.55, veto_win_prob_penalty=0.10
    ).run(_context(), _scores())
    assert out.decision == "open_position"
    assert out.win_probability == 0.62  # 0.72 - 0.10 penalty
    # The veto is still recorded for audit even though it was overridden.
    assert any("skeptic" in v for v in out.vetoes)
    assert out.win_estimates["skeptic"] == 0.15  # raw estimate preserved


def test_soft_veto_penalty_blocks_low_conviction_override() -> None:
    # Two standing vetoes dock the binding win-prob by 0.20, so the PM's 0.70
    # confidence -> 0.50 < 0.55 floor -> reject. (Two vetoes would need PM >= 0.75.)
    client = MockLLMClient(
        [
            "catalyst",
            "options",
            "VETO: dilution\nWIN_PROB: 0.15",  # skeptic vetoes
            "VETO: unbounded risk\nWIN_PROB: 0.15",  # risk vetoes
            _proposal_with_confidence(0.70),
        ]
    )
    out = Committee(
        client, min_win_probability=0.55, veto_win_prob_penalty=0.10
    ).run(_context(), _scores())
    assert out.decision == "reject"
    assert out.win_probability == 0.50  # 0.70 - 2 * 0.10
    assert "0.55" in out.rationale


def test_soft_veto_disabled_keeps_hard_block() -> None:
    # penalty 0 (default) => a single veto is still an absolute block even if the
    # PM is highly confident; the trade never reaches the win-prob math.
    client = MockLLMClient(
        [
            "catalyst",
            "options",
            "VETO: dilution risk\nWIN_PROB: 0.15",
            "acceptable\nWIN_PROB: 0.72",
            _proposal_with_confidence(0.90),
        ]
    )
    out = Committee(client, min_win_probability=0.55).run(_context(), _scores())
    assert out.decision == "reject"
    assert "veto" in out.rationale.lower()


def test_omitted_adversary_falls_back_to_primary() -> None:
    primary = MockLLMClient(
        ["catalyst", "options", "skeptic ok", "risk ok", json.dumps(_PROPOSAL)]
    )
    out = Committee(primary).run(_context(), _scores())
    assert out.decision == "open_position"
    assert len(primary.calls) == 5  # all five roles ran on the one client


def test_usage_total_sums_distinct_clients() -> None:
    primary = MockLLMClient(["catalyst", "options"])
    adversary = MockLLMClient(["skeptic ok", "risk ok"])
    pro = MockLLMClient([json.dumps(_PROPOSAL)])
    committee = Committee(primary, pro_client=pro, adversary_client=adversary)
    committee.run(_context(), _scores())
    assert committee.usage_total().calls == 5  # 2 + 2 + 1


def test_usage_total_dedupes_shared_client() -> None:
    primary = MockLLMClient(["c", "o", "s", "r", json.dumps(_PROPOSAL)])
    committee = Committee(primary)  # one client backs all three tiers
    committee.run(_context(), _scores())
    assert committee.usage_total().calls == 5  # counted once, not tripled
