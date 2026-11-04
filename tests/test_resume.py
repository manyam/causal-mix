import pytest

from causal_mix.common import BudgetStop, Store
from causal_mix.pipeline import require_budget, validate_record


def test_incompatible_resume_rejected(cfg):
    Store(cfg)
    cfg["seed"] = 3
    with pytest.raises(ValueError, match="incompatible resume"):
        Store(cfg)


def test_running_record_is_retried_and_complete_skipped():
    assert not validate_record({"status": "running", "signature": "x"}, "x")
    assert validate_record({"status": "complete", "signature": "x"}, "x")
    with pytest.raises(ValueError):
        validate_record({"status": "complete", "signature": "y"}, "x")


def test_budget_stop_preserves_state(cfg):
    store = Store(cfg)
    store.put("clock.json", {"elapsed": 7190})
    store.start_clock()
    with pytest.raises(BudgetStop):
        require_budget(store, 60, "next run")
    assert store.get("status.json")["state"] == "budget_limited"


def test_changed_dataset_rejected(cfg):
    from causal_mix.data import load_data
    store = Store(cfg)
    store.put("manifest.json", {"data_hashes": {"train": "original"}})
    store.put("data/train.json", {"math": []})
    with pytest.raises(ValueError, match="changed"):
        load_data(store, "train")


def test_failed_synthetic_is_not_reused(cfg):
    from causal_mix.causal import synthetic
    store = Store(cfg)
    store.put("synthetic.json", {"passed": False})
    with pytest.raises(RuntimeError, match="Previous synthetic"):
        synthetic(store)
