import numpy as np
import pytest

from causal_mix.causal import check_treatments, grouped_folds
from causal_mix.common import analytical_policy, draw_mixtures


def test_policies_valid_and_all_negative():
    rng = np.random.default_rng(42)
    t = draw_mixtures(rng, 100)
    assert np.all(t >= .05) and np.allclose(t.sum(1), 1)
    assert np.allclose(analytical_policy([-1, 0, -3]), [1 / 3] * 3)
    positive = analytical_policy([1, 3, -1])
    assert np.all(positive >= .05) and positive.sum() == pytest.approx(1)


def test_grouped_cross_fit():
    groups = np.arange(16).repeat(2)
    x = np.zeros((32, 9))
    seen = []
    for a, b in grouped_folds(x, groups, 4):
        assert not set(groups[a]) & set(groups[b])
        seen.extend(b)
    assert sorted(seen) == list(range(32))


def test_degenerate_treatment_rejected():
    with pytest.raises(ValueError, match="Degenerate"):
        check_treatments(np.zeros((32, 3)), 10000)
