from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from trading_agent.domain.evidence import CandidateContext
from trading_agent.domain.proposals import OpenPositionProposal
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

    def run(self, context: CandidateContext, scores: ScoreComponents) -> CommitteeOutput:
        briefing = self._briefing(context, scores)
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

        catalyst = self.client.complete(system=_CATALYST_SYSTEM, user=briefing)
        options = self.client.complete(system=_OPTIONS_SYSTEM, user=briefing)
        calls += 2

        analyst_context = (
            f"{briefing}\n\n{flag_block}\n\n[catalyst_analyst]\n{catalyst}"
            f"\n\n[options_analyst]\n{options}"
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
            system=_PM_SYSTEM, user=pm_context, json_mode=True
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

        return self._finalize(pm_raw, context, notes, vetoes, calls, red_flags)

    def _finalize(
        self,
        pm_raw: str,
        context: CandidateContext,
        notes: list[RoleNote],
        vetoes: list[str],
        calls: int,
        red_flags: list[RedFlag],
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
