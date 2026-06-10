from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from trading_agent.domain.evidence import CandidateContext
from trading_agent.domain.proposals import OpenPositionProposal, OptionCandidate
from trading_agent.research.llm import LLMClient, LlmUsage
from trading_agent.research.redflags import (
    RedFlag,
    critical_flags,
    detect_red_flags,
    format_red_flags,
)
from trading_agent.research.scoring import ScoreComponents

VETO_PREFIX = "VETO"

# Shared rules every role must respect. Keeps the committee inside the mandate
# and forces evidence discipline (plan sections 6.4 and 12).
_COMMON_RULES = """You are part of an automated research committee for a USD 100
small-cap U.S. options experiment. Hard rules:
- Use only the supplied public evidence. Never invent facts, prices, or filings.
- Label any claim as observed_fact, inference, or speculation.
- Allowed trades: long calls or long puts only, one contract, limit orders.
- Never allege insider trading and never rely on rumors or private information.
- Be concise: at most 6 sentences unless asked for JSON."""

_CATALYST_SYSTEM = (
    _COMMON_RULES
    + "\n\nRole: catalyst_analyst. Explain the public catalyst and its expected"
    " timing. State which evidence_ids support it. Distinguish confirmed dates"
    " from speculation."
)

_OPTIONS_SYSTEM = (
    _COMMON_RULES
    + "\n\nRole: options_analyst. Recommend at most one liquid option contract"
    " (call or put) consistent with the catalyst direction and the mandate"
    " (14-45 DTE, premium small enough for a USD 100 account). Note liquidity"
    " concerns. You do not have a live chain; describe the contract profile you"
    " would want and any data you would need to confirm."
)

# Used instead of _OPTIONS_SYSTEM when the briefing carries a real candidate list
# (sourced from the live option chain and already liquidity-checked). The analyst
# must pick from those contracts rather than describe a hypothetical one.
_OPTIONS_SYSTEM_WITH_CHAIN = (
    _COMMON_RULES
    + "\n\nRole: options_analyst. The briefing lists CANDIDATE CONTRACTS that"
    " already pass the mandate's liquidity, DTE, and cost limits. Recommend"
    " exactly ONE of them, by its exact option_code, consistent with the catalyst"
    " direction. Prefer adequate open interest/volume and a tight spread. Do not"
    " propose any contract that is not in the candidate list."
)

# Appended to the portfolio_manager prompt in candidate mode so the final
# option_code is constrained to a real, tradeable contract.
_PM_CANDIDATE_RULE = (
    "\nThe option_code MUST be copied exactly from one of the CANDIDATE CONTRACTS"
    " in the briefing. Do not invent or modify a contract code."
)

_SKEPTIC_SYSTEM = (
    _COMMON_RULES
    + "\n\nRole: skeptic. Hunt for dilution, ATM shelves, insider selling, stale"
    " news, weak evidence, and IV-crush risk. The briefing includes deterministic"
    " red flags already computed by code; treat them as confirmed facts and weigh"
    " them rather than re-deriving them. If the trade should not proceed, begin"
    f" your reply with '{VETO_PREFIX}:' followed by the reason."
)

_RISK_SYSTEM = (
    _COMMON_RULES
    + "\n\nRole: risk_manager. Argue against the trade when downside is poorly"
    " bounded or the catalyst window is unclear. Weigh the deterministic red flags"
    " in the briefing. If risk is unacceptable, begin your reply with"
    f" '{VETO_PREFIX}:' followed by the reason."
)

_PM_SYSTEM = (
    _COMMON_RULES
    + "\n\nRole: portfolio_manager. Produce the FINAL decision as a single JSON"
    " object and nothing else.\n"
    "If you decide not to trade, output exactly:"
    ' {"decision": "hold"|"reject", "rationale": "<short reason>"}.\n'
    "If you open a position, output an object with these keys:\n"
    '{"decision": "open_position", "ticker", "option_code", "option_side":'
    ' "call"|"put", "action": "buy_to_open", "contracts": 1, "limit_price",'
    ' "max_limit_price", "thesis", "evidence_ids": [..], "confidence": 0..1,'
    ' "expected_catalyst_window": "YYYY-MM-DD/YYYY-MM-DD", "exit_plan":'
    ' {"take_profit_pct", "stop_loss_pct", "time_stop": "YYYY-MM-DD"},'
    ' "invalidation": [..]}.\n'
    "Every evidence_id MUST come from the supplied evidence. If a skeptic or"
    " risk_manager veto stands, do not open a position."
)


class RoleNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    note: str
    model: str | None = None
    vetoed: bool = False


class CommitteeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["open_position", "hold", "reject"]
    rationale: str
    proposal: OpenPositionProposal | None = None
    role_notes: list[RoleNote]
    vetoes: list[str]
    llm_calls: int
    red_flags: list[RedFlag] = []


class Committee:
    """Sequential 5-role pipeline producing a schema-checked decision.

    Three model tiers:
    - ``client`` (fast "flash" model) runs catalyst_analyst and options_analyst;
    - ``adversary_client`` runs the skeptic and risk_manager. Point it at a
      different provider so the adversarial roles' errors are uncorrelated with
      the idea-generating roles. Falls back to ``client`` when omitted;
    - ``pro_client`` (stronger model) runs the decisive portfolio_manager. Falls
      back to ``client`` when omitted.

    The committee never touches the broker or the risk gate. Its output is fed
    into the deterministic risk gate downstream, so invalid JSON or uncited
    evidence here results in a reject rather than an order.
    """

    def __init__(
        self,
        client: LLMClient,
        pro_client: LLMClient | None = None,
        adversary_client: LLMClient | None = None,
    ):
        self.client = client
        self.pro_client = pro_client or client
        self.adversary_client = adversary_client or client

    def usage_total(self) -> LlmUsage:
        """Aggregate token/call usage across the distinct clients in use.

        Snapshot this before and after ``run`` (see ``llm.usage_delta``) to meter
        the spend of a single committee pass.
        """

        total = LlmUsage()
        seen: list[LLMClient] = []
        for client in (self.client, self.adversary_client, self.pro_client):
            if any(client is other for other in seen):
                continue
            seen.append(client)
            usage = getattr(client, "usage", None)
            if usage is not None:
                total.add(usage)
        return total

    def run(
        self,
        context: CandidateContext,
        scores: ScoreComponents,
        candidates: list[OptionCandidate] | None = None,
        market_snapshot: dict | None = None,
    ) -> CommitteeOutput:
        # In candidate mode the options_analyst/PM pick a real listed contract
        # from ``candidates`` instead of guessing one; the PM's option_code is
        # then constrained to that set. ``candidates=None`` keeps legacy behavior.
        candidate_block = self._format_candidates(candidates)
        candidate_codes = (
            {c.option_code for c in candidates} if candidates else None
        )
        options_system = _OPTIONS_SYSTEM_WITH_CHAIN if candidate_block else _OPTIONS_SYSTEM
        pm_system = _PM_SYSTEM + (_PM_CANDIDATE_RULE if candidate_block else "")

        briefing = self._briefing(context, scores)
        snapshot_block = self._format_snapshot(market_snapshot)
        if snapshot_block:
            briefing = f"{briefing}\n\n{snapshot_block}"
        flash_model = self._model_name(self.client)
        adversary_model = self._model_name(self.adversary_client)
        pro_model = self._model_name(self.pro_client)

        red_flags = detect_red_flags(context, scores)
        criticals = critical_flags(red_flags)
        # A critical, code-detected red flag (a fresh dilution shelf or insider
        # selling) blocks the trade deterministically and skips the committee
        # entirely: a known disqualifier never depends on the LLM noticing it,
        # and the block costs zero API calls.
        if criticals:
            vetoes = [f"redflags: {flag.message}" for flag in criticals]
            return CommitteeOutput(
                decision="reject",
                rationale="Blocked by deterministic red flag(s): "
                + ", ".join(flag.code for flag in criticals),
                proposal=None,
                role_notes=[],
                vetoes=vetoes,
                llm_calls=0,
                red_flags=red_flags,
            )

        flag_block = format_red_flags(red_flags)
        calls = 0

        options_briefing = (
            f"{briefing}\n\n{candidate_block}" if candidate_block else briefing
        )
        catalyst = self.client.complete(system=_CATALYST_SYSTEM, user=briefing)
        options = self.client.complete(system=options_system, user=options_briefing)
        calls += 2

        analyst_context = (
            f"{briefing}\n\n{flag_block}"
            + (f"\n\n{candidate_block}" if candidate_block else "")
            + f"\n\n[catalyst_analyst]\n{catalyst}\n\n[options_analyst]\n{options}"
        )
        skeptic = self.adversary_client.complete(
            system=_SKEPTIC_SYSTEM, user=analyst_context
        )
        risk = self.adversary_client.complete(
            system=_RISK_SYSTEM, user=f"{analyst_context}\n\n[skeptic]\n{skeptic}"
        )
        calls += 2

        notes = [
            RoleNote(role="catalyst_analyst", note=catalyst.strip(), model=flash_model),
            RoleNote(role="options_analyst", note=options.strip(), model=flash_model),
            RoleNote(
                role="skeptic",
                note=skeptic.strip(),
                model=adversary_model,
                vetoed=self._is_veto(skeptic),
            ),
            RoleNote(
                role="risk_manager",
                note=risk.strip(),
                model=adversary_model,
                vetoed=self._is_veto(risk),
            ),
        ]
        vetoes = [f"{note.role}: {note.note}" for note in notes if note.vetoed]

        pm_context = (
            f"{analyst_context}\n\n[skeptic]\n{skeptic}\n\n[risk_manager]\n{risk}"
        )
        pm_raw = self.pro_client.complete(
            system=pm_system, user=pm_context, json_mode=True
        )
        calls += 1
        notes.append(
            RoleNote(role="portfolio_manager", note=pm_raw.strip(), model=pro_model)
        )

        # A standing veto blocks any open decision regardless of the PM's vote.
        if vetoes:
            return CommitteeOutput(
                decision="reject",
                rationale="Blocked by committee veto: " + "; ".join(vetoes),
                proposal=None,
                role_notes=notes,
                vetoes=vetoes,
                llm_calls=calls,
                red_flags=red_flags,
            )

        return self._finalize(
            pm_raw, context, notes, vetoes, calls, red_flags, candidate_codes
        )

    def _finalize(
        self,
        pm_raw: str,
        context: CandidateContext,
        notes: list[RoleNote],
        vetoes: list[str],
        calls: int,
        red_flags: list[RedFlag],
        candidate_codes: set[str] | None = None,
    ) -> CommitteeOutput:
        def reject(rationale: str) -> CommitteeOutput:
            return CommitteeOutput(
                decision="reject",
                rationale=rationale,
                proposal=None,
                role_notes=notes,
                vetoes=vetoes,
                llm_calls=calls,
                red_flags=red_flags,
            )

        try:
            payload = json.loads(self._strip_fences(pm_raw))
        except json.JSONDecodeError as exc:
            return reject(f"portfolio_manager returned invalid JSON: {exc}")
        if not isinstance(payload, dict):
            return reject("portfolio_manager JSON was not an object")

        decision = payload.get("decision")
        if decision in {"hold", "reject"}:
            return CommitteeOutput(
                decision=decision,
                rationale=str(payload.get("rationale", "")).strip()
                or "portfolio_manager declined to trade",
                proposal=None,
                role_notes=notes,
                vetoes=vetoes,
                llm_calls=calls,
                red_flags=red_flags,
            )
        if decision != "open_position":
            return reject(f"portfolio_manager returned unknown decision: {decision!r}")

        try:
            proposal = OpenPositionProposal.model_validate(payload)
        except ValidationError as exc:
            return reject(f"proposal failed schema validation: {exc}")

        if proposal.ticker != context.ticker:
            return reject(
                f"proposal ticker {proposal.ticker!r} does not match candidate "
                f"{context.ticker!r}"
            )

        allowed_ids = context.evidence_ids()
        unknown = [eid for eid in proposal.evidence_ids if eid not in allowed_ids]
        if unknown:
            return reject(f"proposal cites unknown evidence_ids: {unknown}")

        if candidate_codes is not None and proposal.option_code not in candidate_codes:
            return reject(
                f"proposal option_code {proposal.option_code!r} is not in the "
                "candidate contract list"
            )

        return CommitteeOutput(
            decision="open_position",
            rationale=proposal.thesis,
            proposal=proposal,
            role_notes=notes,
            vetoes=vetoes,
            llm_calls=calls,
            red_flags=red_flags,
        )

    @staticmethod
    def _briefing(context: CandidateContext, scores: ScoreComponents) -> str:
        lines = [
            f"Ticker: {context.ticker}",
            f"As of: {context.as_of.isoformat()}",
            "",
            "Deterministic score components (computed before this committee):",
            f"  catalyst={scores.catalyst} operations={scores.operations} "
            f"contradictions={scores.contradictions} total={scores.total}",
            "",
            "Public evidence:",
        ]
        for item in context.evidence:
            lines.append(
                f"  [{item.evidence_id}] ({item.source_type}, "
                f"published {item.published_at.isoformat()}) {item.observed_fact} "
                f"<{item.source_url}>"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_snapshot(snapshot: dict | None) -> str:
        if not snapshot or not snapshot.get("price"):
            return ""
        return (
            "Underlying market snapshot (delayed ~15min; observed_fact quality):\n"
            f"  price={snapshot.get('price')} "
            f"day_change={snapshot.get('change_pct', 0):+.2f}% "
            f"prev_close={snapshot.get('prev_close')} "
            f"day_range={snapshot.get('day_low')}-{snapshot.get('day_high')} "
            f"volume={snapshot.get('volume')} "
            f"iv30={snapshot.get('iv30')} (chg {snapshot.get('iv30_change', 0):+.2f})"
        )

    @staticmethod
    def _format_candidates(candidates: list[OptionCandidate] | None) -> str:
        if not candidates:
            return ""
        lines = [
            "CANDIDATE CONTRACTS (already pass the mandate's liquidity/DTE/cost "
            "limits; choose option_code from THIS list only):"
        ]
        for c in candidates:
            lines.append(
                f"  {c.option_code} {c.option_side} strike={c.strike} "
                f"expiry={c.expiry.isoformat()} DTE={c.dte} bid={c.bid} ask={c.ask} "
                f"OI={c.open_interest} vol={c.daily_volume} iv={c.iv} "
                f"est_cost=${c.estimated_contract_cost_usd}"
            )
        return "\n".join(lines)

    @staticmethod
    def _model_name(client: LLMClient) -> str | None:
        return getattr(client, "model", None)

    @staticmethod
    def _is_veto(text: str) -> bool:
        return text.strip().upper().startswith(VETO_PREFIX)

    @staticmethod
    def _strip_fences(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]  # drop opening fence (``` or ```json)
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text
