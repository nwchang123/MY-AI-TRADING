from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Deterministic category weights. Only categories that are actually computed
# from the available data are scored. ``options`` and ``underlying`` (live option
# chain / underlying market data) were never wired into the autonomous cycle, so
# they are intentionally absent rather than silently fed as 0.0 -- re-add them
# here with real market-data inputs once the OpenD quote feed is verified.
# Contradictions are a penalty, so the weight is negative.
CATALYST_WEIGHT = 50.0
OPERATIONS_WEIGHT = 25.0
CONTRADICTION_WEIGHT = -25.0


class ScoreInputs(BaseModel):
    """Normalized 0..1 sub-scores computed before any LLM call.

    Each field is a deterministic feature strength in [0, 1]. ``contradictions``
    measures how strong the disqualifying signals are (1.0 = severe dilution or
    insider selling), so a higher value lowers the total.
    """

    model_config = ConfigDict(extra="forbid")

    catalyst: float = Field(ge=0, le=1)
    operations: float = Field(ge=0, le=1)
    contradictions: float = Field(ge=0, le=1)


class ScoreComponents(BaseModel):
    """Weighted score breakdown handed to the committee as context."""

    model_config = ConfigDict(extra="forbid")

    catalyst: float
    operations: float
    contradictions: float
    total: float


def score_candidate(inputs: ScoreInputs) -> ScoreComponents:
    catalyst = inputs.catalyst * CATALYST_WEIGHT
    operations = inputs.operations * OPERATIONS_WEIGHT
    contradictions = inputs.contradictions * CONTRADICTION_WEIGHT
    total = catalyst + operations + contradictions
    return ScoreComponents(
        catalyst=round(catalyst, 4),
        operations=round(operations, 4),
        contradictions=round(contradictions, 4),
        total=round(total, 4),
    )
