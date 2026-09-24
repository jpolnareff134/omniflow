import pytest

from eval.budget_matching import select_by_sample_ratio


def _candidate(index, ratio, peak_recall):
    return {
        "sweep_index": index,
        "report": {
            "sample_ratio": ratio,
            "peak_recall_top5": peak_recall,
        },
    }


def test_selection_uses_closest_sample_ratio_not_peak_recall():
    candidates = [
        _candidate(0, 0.41, 0.10),
        _candidate(1, 0.43, 0.99),
        _candidate(2, 0.38, 1.00),
    ]

    selection = select_by_sample_ratio(candidates, target=0.40, slack=0.05)

    assert selection["selected"] is candidates[0]
    assert selection["n_candidates_in_band"] == 2


def test_selection_does_not_fall_back_outside_requested_band():
    candidates = [
        _candidate(0, 0.39, 0.10),
        _candidate(1, 0.47, 0.99),
    ]

    selection = select_by_sample_ratio(candidates, target=0.40, slack=0.05)

    assert selection["selected"] is None
    assert selection["nearest_below"] is candidates[0]
    assert selection["nearest_above"] is candidates[1]


def test_negative_slack_is_rejected():
    with pytest.raises(ValueError, match="slack must be non-negative"):
        select_by_sample_ratio([], target=0.4, slack=-0.01)
