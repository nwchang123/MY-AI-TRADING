from trading_agent.research.scoring import ScoreInputs, score_candidate


def test_full_strength_inputs_sum_weights() -> None:
    result = score_candidate(
        ScoreInputs(catalyst=1, options=1, underlying=1, operations=1, contradictions=0)
    )
    assert result.catalyst == 30
    assert result.options == 25
    assert result.underlying == 15
    assert result.operations == 15
    assert result.contradictions == 0
    assert result.total == 85


def test_contradictions_reduce_total() -> None:
    clean = score_candidate(
        ScoreInputs(catalyst=0.8, options=0.6, underlying=0.5, operations=0.6, contradictions=0)
    )
    dirty = score_candidate(
        ScoreInputs(catalyst=0.8, options=0.6, underlying=0.5, operations=0.6, contradictions=1)
    )
    assert dirty.contradictions == -25
    assert dirty.total == clean.total - 25
