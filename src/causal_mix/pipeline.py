from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import BudgetStop, DOMAINS, POLICIES, digest, draw_mixtures, progress
from .data import domain_features, load_data, make_pools


def require_budget(store, estimate, stage):
    if not store.budget_available(estimate):
        store.put("status.json", {"state": "budget_limited", "stage": stage,
                  "elapsed_seconds": store.checkpoint_clock(), "next_estimate_seconds": estimate})
        raise BudgetStop(f"Budget insufficient for {stage}; saved progress is resumable")


def benchmark(store):
    from .lm import LanguageModel
    if store.exists("profile.json"):
        profile = store.get("profile.json")
        if profile["selected"] is None:
            raise BudgetStop("No profile fits the budget; see benchmark.json and profile.json")
        progress(f"Using frozen profile: {profile['selected']}")
        return
    train, dev = load_data(store, "train"), load_data(store, "dev")
    timings = store.get("benchmark.json") if store.exists("benchmark.json") else {}
    lengths = sorted({p["length"] for p in store.cfg["budget"]["profiles"]}, reverse=True)
    for role in ("proxy", "transfer"):
        for length in lengths:
            key = f"{role}_{length}"
            if key in timings:
                continue
            require_budget(store, 0, f"benchmark {key}")
            progress(f"Benchmarking {role}, FP32, sequence length {length}")
            start = time.monotonic()
            lm = LanguageModel(store, role, length)
            load_seconds = time.monotonic() - start
            try:
                lm.train(train, [1 / 3] * 3, 1, store.cfg["seed"])
                sample = lm.train(train, [1 / 3] * 3, 2, store.cfg["seed"])
                lm.reset()
                start = time.monotonic()
                baseline = lm.evaluate(dev)
                eval_seconds = time.monotonic() - start
                timings[key] = {"step_seconds": sample["seconds"] / 2,
                                "dev_seconds": eval_seconds, "load_seconds": load_seconds,
                                "truncation": lm.truncation()}
                store.put(f"baselines/{key}_dev.json", baseline)
                store.put("benchmark.json", timings)
                progress(f"{key}: {sample['seconds']/2:.2f}s/step, {eval_seconds:.2f}s/dev evaluation")
            finally:
                lm.close()
    estimates = []
    elapsed = store.checkpoint_clock()
    for profile in store.cfg["budget"]["profiles"]:
        p = timings[f"proxy_{profile['length']}"]
        q = timings[f"transfer_{profile['length']}"]
        feature_seconds = p["dev_seconds"] * store.cfg["data"]["train"] / store.cfg["data"]["dev"]
        collection = profile["runs"] * (profile["steps"] * p["step_seconds"] + p["dev_seconds"] + 1)
        comparison = sum(5 * (profile["steps"] * a["step_seconds"] + 2 * a["dev_seconds"] + 1)
                         + 2 * a["dev_seconds"] + a["load_seconds"] for a in (p, q))
        remaining = feature_seconds + collection + comparison + p["load_seconds"] + 90
        projected = elapsed + remaining / (1 - store.cfg["budget"]["headroom"])
        estimates.append({"profile": profile, "projected_total_seconds": projected,
                          "comparison_reserve_seconds": comparison / .8 + 30})
    selected = next((e for e in estimates if e["projected_total_seconds"] <= store.cfg["budget"]["minutes"] * 60), None)
    store.put("profile.json", {"selected": selected["profile"] if selected else None,
              "estimates": estimates, "frozen_before_outcomes": True})
    if selected is None:
        raise BudgetStop("Even the smallest profile exceeds the time budget; no proxy run was started")
    progress(f"Selected {selected['profile']}; projected total {selected['projected_total_seconds']/60:.1f} minutes")


def selected_profile(store):
    profile = store.get("profile.json")["selected"]
    if profile is None:
        raise BudgetStop("No profile fits the budget")
    return profile


def validate_record(record, signature):
    if record.get("signature") != signature:
        raise ValueError("Run identity differs from current experiment; refusing incompatible resume")
    return record.get("status") == "complete"


def collect(store):
    from .lm import LanguageModel, improvement, mean_losses
    if not store.exists("synthetic.json") or not store.get("synthetic.json")["passed"]:
        raise ValueError("Run synthetic validation successfully before collecting proxy outcomes")
    profile = selected_profile(store)
    signature = store.signature()
    train, dev = load_data(store, "train"), load_data(store, "dev")
    expected = [f"runs/proxy_{i:03d}.json" for i in range(profile["runs"])]
    if all(store.exists(n) and validate_record(store.get(n), signature) for n in expected):
        progress("All proxy runs already complete")
        return
    lm = LanguageModel(store, "proxy", profile["length"])
    try:
        baseline = store.get(f"baselines/proxy_{profile['length']}_dev.json")
        if not store.exists("pools.json"):
            require_budget(store, 0, "pool features")
            lm.reset()
            progress("Scoring training examples with the frozen proxy model")
            losses = lm.evaluate(train)
            store.put("training_losses.json", losses)
            pools = make_pools(train, losses, store.cfg, profile["runs"] // 2)
            target = sum([domain_features(train[d], losses[d]) for d in DOMAINS], [])
            store.put("pools.json", {"signature": signature, "pools": pools, "target_x": target})
        state = store.get("pools.json")
        if state["signature"] != signature:
            raise ValueError("Pool configuration mismatch")
        pools = state["pools"]
        by_id = {r["id"]: r for d in DOMAINS for r in train[d]}
        rng = np.random.default_rng(np.random.SeedSequence([store.cfg["seed"], 300]))
        mixtures = draw_mixtures(rng, profile["runs"], store.cfg["estimation"]["floor"])
        timing = store.get("benchmark.json")[f"proxy_{profile['length']}"]
        estimate = (profile["steps"] * timing["step_seconds"] + timing["dev_seconds"]) / .8
        reserve = next(e["comparison_reserve_seconds"] for e in store.get("profile.json")["estimates"] if e["profile"] == profile)
        for i, mixture in enumerate(mixtures):
            name = expected[i]
            if store.exists(name) and validate_record(store.get(name), signature):
                continue
            require_budget(store, estimate + reserve, f"proxy run {i + 1}/{profile['runs']}")
            pool = pools[i // 2]
            rows = {d: [by_id[key] for key in pool["members"][d]] for d in DOMAINS}
            record = {"signature": signature, "status": "running", "run_id": i,
                      "pool_id": pool["pool_id"], "x": pool["x"], "t": mixture.tolist(),
                      "sampling_seed": store.cfg["seed"] + i, "profile": profile}
            store.put(name, record)
            start = time.monotonic()
            training = lm.train(rows, mixture, profile["steps"], record["sampling_seed"])
            losses = lm.evaluate(dev)
            record.update(status="complete", y=improvement(baseline, losses),
                          domain_nll=mean_losses(losses), training=training,
                          seconds=time.monotonic() - start, truncation=lm.truncation())
            store.put(name, record)
            store.checkpoint_clock()
            progress(f"Proxy {i+1}/{profile['runs']}: development improvement {record['y']:+.5f}")
    finally:
        lm.close()


def fit(store):
    from .causal import estimate
    profile = selected_profile(store)
    signature = store.signature()
    if store.exists("policies.json"):
        if store.get("policies.json")["signature"] != signature:
            raise ValueError("Policy configuration mismatch")
        return
    runs = []
    for i in range(profile["runs"]):
        name = f"runs/proxy_{i:03d}.json"
        if not store.exists(name) or not validate_record(store.get(name), signature):
            raise ValueError("All planned proxy runs must complete before causal fitting")
        runs.append(store.get(name))
    state = store.get("pools.json")
    require_budget(store, 30, "causal fitting")
    progress("Fitting causal forest and nested grouped diagnostics")
    result, _ = estimate([r["x"] for r in runs], [r["t"] for r in runs], [r["y"] for r in runs],
                         [r["pool_id"] for r in runs], state["target_x"], store.cfg)
    result["signature"] = signature
    result["selection_split"] = "dev"
    result["frozen_before_test"] = True
    store.put("policies.json", result)
    for warning in result["warnings"]:
        progress("Diagnostic: " + warning)


def compare(store):
    from .lm import LanguageModel, improvement, mean_losses
    profile = selected_profile(store)
    fixed = store.get("policies.json")  # Must exist before the test file is read.
    if fixed["signature"] != store.signature():
        raise ValueError("Frozen policies do not match current configuration")
    signature = digest({"experiment": store.signature(), "policies": fixed})
    train, dev, test = (load_data(store, split) for split in ("train", "dev", "test"))
    timings = store.get("benchmark.json")
    for role in ("proxy", "transfer"):
        names = [f"comparisons/{role}_{p}.json" for p in POLICIES]
        if all(store.exists(n) and validate_record(store.get(n), signature) for n in names):
            continue
        lm = LanguageModel(store, role, profile["length"])
        try:
            timing = timings[f"{role}_{profile['length']}"]
            require_budget(store, timing["dev_seconds"], f"{role} test baseline")
            lm.reset()
            baseline_dev = store.get(f"baselines/{role}_{profile['length']}_dev.json")
            baseline_name = f"baselines/{role}_{profile['length']}_test.json"
            if not store.exists(baseline_name):
                store.put(baseline_name, lm.evaluate(test))
            baseline_test = store.get(baseline_name)
            for policy, name in zip(POLICIES, names):
                if store.exists(name) and validate_record(store.get(name), signature):
                    continue
                estimate = (profile["steps"] * timing["step_seconds"] + 2 * timing["dev_seconds"]) / .8
                require_budget(store, estimate, f"{role} {policy} comparison")
                record = {"signature": signature, "role": role, "policy": policy,
                          "mixture": fixed["policies"][policy], "status": "running", "profile": profile}
                store.put(name, record)
                progress(f"Comparing {role}: {policy}")
                training = lm.train(train, record["mixture"], profile["steps"], store.cfg["seed"])
                dev_losses, test_losses = lm.evaluate(dev), lm.evaluate(test)
                adapter_path = store.path(f"adapters/{role}/{policy}")
                lm.save(adapter_path)
                record.update(status="complete", training=training, truncation=lm.truncation(),
                              adapter_path=str(adapter_path), dev_improvement=improvement(baseline_dev, dev_losses),
                              test_improvement=improvement(baseline_test, test_losses),
                              dev_nll=mean_losses(dev_losses), test_nll=mean_losses(test_losses),
                              baseline_test_nll=mean_losses(baseline_test), test_example_nll=test_losses)
                store.put(name, record)
                store.checkpoint_clock()
                progress(f"{role}/{policy}: test improvement {record['test_improvement']:+.5f}")
        finally:
            lm.close()
    store.put("status.json", {"state": "complete", "elapsed_seconds": store.checkpoint_clock(),
              "comparison_runs": 10, "proxy_runs": profile["runs"]})
