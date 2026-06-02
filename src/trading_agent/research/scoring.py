from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Deterministic category weights from DEVELOPMENT_PLAN.md section 6.3.
# Contradictions are a penalty, so the weight is negative.
CATALYST_WEIGHT = 30.0
OPTIONS_WEIGHT = 25.0
UNDERLYING_WEIGHT = 15.0
OPERATIONS_WEIGHT = 15.0
CONTRADICTION_WEIGHT = -25.0


class ScoreInputs(BaseModel):
    """Normalized 0..1 sub-scores computed before any LLM call.

    Each field is a deterministic feature strength in [0, 1]. ``contradictions``
    measures how strong the disqualifying signals are (1.0 = severe dilution,
    insider selling, or poor liquidity), so a higher value lowers the total.
    """

    model_config = ConfigDict(extra="forbid")

    catalyst: float = Field(ge=0, le=1)
    options: float = Field(ge=0, le=1)
    underlying: float = Field(ge=0, le=1)
    operations: float = Field(ge=0, le=1)
    contradictions: float = Field(ge=0, le=1)


class ScoreComponents(BaseModel):
    """Weighted score breakdown handed to the committee as context."""

    model_config = ConfigDict(extra="forbid")

    catalyst: float
    options: float
    underlying: float
    operations: float
    contradictions: float
    total: float


def score_candidate(inputs: ScoreInputs) -> ScoreComponents:
    catalyst = inputs.catalyst * CATALYST_WEIGHT
    options = inputs.options * OPTIONS_WEIGHT
    underlying = inputs.underlying * UNDERLYING_WEIGHT
    operations = inputs.operations * OPERATIONS_WEIGHT
    contradictions = inputs.contradictions * CONTRADICTION_WEIGHT
    total = catalyst + options + underlying + operations + contradictions
    return ScoreComponents(
        catalyst=round(catalyst, 4),
        options=round(options, 4),
        underlying=round(underlying, 4),
        operations=round(operations, 4),
        contradictions=round(contradictions, 4),
        total=round(total, 4),
    )
