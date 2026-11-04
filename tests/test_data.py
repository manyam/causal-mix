import numpy as np

from causal_mix.common import DOMAINS, digest
from causal_mix.data import adapt_row, make_pools, normalize_prompt, select_splits


def row(d, prompt, n):
    return {"id": f"{d}:{n}", "prompt_hash": digest(normalize_prompt(prompt)), "answer": str(n) + " answer"}


def test_split_isolation_reserves_all_official_test_prompts():
    raw = {d: {"train": [row(d, f"{d} train {i}", i) for i in range(12)],
               "test": [row(d, f"{d} test {i}", 100 + i) for i in range(4)]} for d in DOMAINS}
    raw["math"]["train"].insert(0, row("math", "coding test 3", 200))
    raw["math"]["train"].insert(0, row("math", "MATH   TEST 2", 201))
    chosen = select_splits(raw, {"train": 5, "dev": 2, "test": 2}, 42)
    hashes = [{r["prompt_hash"] for rows in chosen[s].values() for r in rows} for s in chosen]
    assert not hashes[0] & hashes[1] and not hashes[0] & hashes[2] and not hashes[1] & hashes[2]
    heldout = {r["prompt_hash"] for d in DOMAINS for r in raw[d]["test"]}
    assert not heldout & hashes[0] and not heldout & hashes[1]
    assert chosen == select_splits(raw, {"train": 5, "dev": 2, "test": 2}, 42)


def test_first_assistant_turn_only():
    r = adapt_row("instructions", {"prompt_id": "a", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]}, "train")
    assert r["prompt"] == "u" and r["answer"] == "a"


def test_pool_covariates_do_not_depend_on_treatment(cfg):
    cfg["data"]["pool_per_domain"] = 8
    rows = {d: [row(d, f"{d} {i}", i) for i in range(20)] for d in DOMAINS}
    losses = {d: np.linspace(1, 3, 20) for d in DOMAINS}
    first = make_pools(rows, losses, cfg, 4)
    np.random.default_rng(9).dirichlet([1, 1, 1], size=100)
    second = make_pools(rows, losses, cfg, 4)
    assert first == second
    assert len(first[0]["x"]) == 9
    assert first[0]["members"] != first[1]["members"]
