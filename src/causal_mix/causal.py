from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor

from .common import analytical_policy, draw_mixtures, progress


def nuisance(seed):
    return LGBMRegressor(n_estimators=30, num_leaves=4, max_depth=2, min_child_samples=5,
                         learning_rate=.05, reg_lambda=1., verbosity=-1, n_jobs=1, random_state=seed)


def grouped_folds(x, groups, count):
    if len(np.unique(groups)) < count:
        raise ValueError("Too few independent pools for grouped cross-fitting")
    return list(GroupKFold(n_splits=count).split(x, groups=groups))


def check_treatments(z, threshold):
    centered = z - z.mean(axis=0)
    rank = np.linalg.matrix_rank(centered)
    condition = float(np.linalg.cond(centered))
    if rank < z.shape[1] or not np.isfinite(condition):
        raise ValueError("Degenerate treatment variation: log-mixture design is rank deficient")
    return {"rank": int(rank), "condition": condition, "ill_conditioned": condition > threshold}


def forest(cfg, folds, seed):
    from econml.dml import CausalForestDML
    ec = cfg["estimation"]
    return CausalForestDML(model_y=nuisance(seed), model_t=MultiOutputRegressor(nuisance(seed)),
                          cv=folds, discrete_treatment=False, n_estimators=ec["trees"],
                          max_depth=ec["max_depth"], min_samples_leaf=ec["min_samples_leaf"],
                          max_samples=1., honest=True, inference=False,
                          min_var_fraction_leaf=.01, n_jobs=1, random_state=seed)


def estimate(x, t, y, groups, target_x, cfg, diagnostics=True):
    ec, seed = cfg["estimation"], cfg["seed"]
    x, t, y, groups = map(np.asarray, (x, t, y, groups))
    if not all(np.isfinite(a).all() for a in (x, t, y, target_x)):
        raise ValueError("Non-finite meta-data")
    if np.any(t <= 0) or not np.allclose(t.sum(1), 1.):
        raise ValueError("Mixtures must be positive and sum to one")
    z = np.log(t + ec["epsilon"])
    condition = check_treatments(z, ec["condition_threshold"])
    folds = grouped_folds(x, groups, ec["folds"])
    learner = forest(cfg, folds, seed)
    learner.fit(y, z, X=x)
    theta = np.asarray(learner.const_marginal_effect(np.asarray(target_x).reshape(1, -1))).reshape(-1)
    if not np.isfinite(theta).all():
        raise ValueError("Non-finite marginal effects")
    rng = np.random.default_rng(seed + 200)
    candidates = draw_mixtures(rng, ec["candidates"], ec["floor"])
    gain = (np.log(candidates + ec["epsilon"]) - np.log(1 / 3 + ec["epsilon"])) @ theta
    top = np.argsort(gain)[-ec["top_k"]:]
    regmix = nuisance(seed)
    regmix.fit(z, y)
    reg_scores = regmix.predict(np.log(candidates + ec["epsilon"]))
    policies = {"uniform": [1 / 3] * 3, "best_proxy": t[np.argmax(y)].tolist(),
                "regmix": candidates[np.argsort(reg_scores)[-ec["top_k"]:]].mean(0).tolist(),
                "causalmix_a": analytical_policy(theta, ec["floor"]).tolist(),
                "causalmix_s": candidates[top].mean(0).tolist()}
    result = {"theta": theta.tolist(), "policies": policies, "conditioning": condition,
              "samples": len(y), "independent_pools": len(np.unique(groups)),
              "warnings": [], "fold_groups": [{"train": np.unique(groups[a]).tolist(),
                  "validation": np.unique(groups[b]).tolist()} for a, b in folds]}
    if len(y) < 100:
        result["warnings"].append("Very few proxy runs: heterogeneous effects and policy rankings are preliminary.")
    if condition["ill_conditioned"]:
        result["warnings"].append("Treatment design is poorly conditioned.")
    if np.all(theta <= 0):
        result["warnings"].append("All estimated effects are non-positive; analytical policy defaults to uniform.")
    if diagnostics:
        # Outer grouped folds keep effect fitting AND nuisance estimation away from evaluation pools.
        residual_y = np.empty_like(y, dtype=float)
        residual_z = np.empty_like(z, dtype=float)
        predicted = np.empty_like(y, dtype=float)
        constant = np.empty_like(y, dtype=float)
        for fold_index, (a, b) in enumerate(folds):
            ny, nt = nuisance(seed), MultiOutputRegressor(nuisance(seed))
            ny.fit(x[a], y[a]); nt.fit(x[a], z[a])
            residual_y[b] = y[b] - ny.predict(x[b])
            residual_z[b] = z[b] - nt.predict(x[b])
            inner = grouped_folds(x[a], groups[a], min(ec["folds"], len(np.unique(groups[a]))))
            fitted = forest(cfg, inner, seed + fold_index)
            fitted.fit(y[a], z[a], X=x[a], cache_values=True)
            effect = fitted.const_marginal_effect(x[b])
            predicted[b] = np.sum(effect * residual_z[b], axis=1)
            ry, rz, _, _ = fitted.residuals_
            coefficient = np.linalg.solve(rz.T @ rz + 1e-3 * np.eye(3), rz.T @ ry)
            constant[b] = residual_z[b] @ coefficient
        mse = float(np.mean((residual_y - predicted) ** 2))
        zero_mse = float(np.mean(residual_y ** 2))
        result["diagnostics"] = {"outer_grouped_r_loss": mse, "zero_effect_r_loss": zero_mse,
                  "constant_effect_r_loss": float(np.mean((residual_y - constant) ** 2)),
                  "r_score_vs_zero": 1 - mse / max(zero_mse, 1e-12),
                  "residual_treatment_condition": check_treatments(residual_z, ec["condition_threshold"])}
        if mse >= zero_mse:
            result["warnings"].append("Held-out residual prediction does not beat zero effects; do not infer a reliable gain.")
    return result, learner


def synthetic(store):
    if store.exists("synthetic.json"):
        result = store.get("synthetic.json")
        if not result["passed"]:
            raise RuntimeError("Previous synthetic validation failed; refusing to continue")
        return result
    cfg = store.cfg
    rng = np.random.default_rng(cfg["seed"])
    # Two state strata, independent randomized treatments, and paired observations per pool.
    n = 2048
    state = rng.choice([-1., 1.], size=n // 2).repeat(2)
    x = np.column_stack([state, np.zeros((n, 8))])
    truth = np.column_stack([1.5 + .8 * state, 1.5 - .8 * state, np.full(n, .3)])
    t = draw_mixtures(rng, n, cfg["estimation"]["floor"])
    y = 2 * state + np.sum(truth * np.log(t + cfg["estimation"]["epsilon"]), axis=1) + rng.normal(0, .03, n)
    target = np.array([1.] + [0.] * 8)
    fitted, learner = estimate(x, t, y, np.arange(n // 2).repeat(2), target, cfg, diagnostics=False)
    test_x = np.array([[-1.] + [0.] * 8, [1.] + [0.] * 8])
    expected = np.array([[.7, 2.3, .3], [2.3, .7, .3]])
    actual = learner.const_marginal_effect(test_x)
    rmse = float(np.sqrt(np.mean((expected - actual) ** 2)))
    eps = cfg["estimation"]["epsilon"]
    true_gain = {name: float(expected[1] @ (np.log(np.array(t) + eps) - np.log(1 / 3 + eps)))
                 for name, t in fitted["policies"].items()}
    passed = rmse < .5 and true_gain["causalmix_a"] > .1 and true_gain["causalmix_s"] > .1
    result = {"passed": bool(passed), "effect_rmse": rmse, "true_policy_gains": true_gain,
              "expected_effects": expected.tolist(), "estimated_effects": actual.tolist(),
              "description": "Known two-state synthetic response, not LM or paper benchmark evidence"}
    store.put("synthetic.json", result)
    progress(f"Synthetic validation: {'PASS' if passed else 'FAIL'}, effect RMSE={rmse:.4f}")
    if not passed:
        raise RuntimeError("Synthetic recovery failed; inspect synthetic.json before proceeding")
    return result
