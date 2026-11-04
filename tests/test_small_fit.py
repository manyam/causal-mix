"""Exercise the real nested estimator at the demo's small meta-dataset size."""
import numpy as np

from causal_mix.causal import estimate
from causal_mix.common import draw_mixtures


def test_small_grouped_estimator_pipeline(cfg):
    rng = np.random.default_rng(19)
    groups = np.arange(32).repeat(2)
    x = rng.normal(size=(32, 9)).repeat(2, axis=0)
    t = draw_mixtures(rng, 64)
    y = .2 * x[:, 0] + np.log(t) @ np.array([.1, .2, .3]) + rng.normal(0, .01, 64)
    result, _ = estimate(x, t, y, groups, np.zeros(9), cfg)
    assert result["independent_pools"] == 32
    assert np.isfinite(result["diagnostics"]["outer_grouped_r_loss"])
    for mixture in result["policies"].values():
        assert np.all(np.array(mixture) >= .05) and np.isclose(sum(mixture), 1)
    for fold in result["fold_groups"]:
        assert not set(fold["train"]) & set(fold["validation"])
