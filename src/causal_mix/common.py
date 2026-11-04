from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

DOMAINS = ("math", "coding", "instructions")
POLICIES = ("uniform", "best_proxy", "regmix", "causalmix_a", "causalmix_s")


class BudgetStop(RuntimeError):
    """A resumable experiment stopped at its configured compute budget."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    if not 0 < cfg["estimation"]["floor"] < 1 / 3:
        raise ValueError("Domain floor must be between zero and 1/3")
    if any(p["runs"] % 2 or p["runs"] < 16 for p in cfg["budget"]["profiles"]):
        raise ValueError("Profiles need an even number of runs, at least 16")
    return cfg


class Store:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg["artifact_dir"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = digest(cfg)
        state = self.root / "config.json"
        if state.exists():
            if digest(read_json(state)) != self.config_hash:
                raise ValueError("Configuration changed: use a new artifact_dir; refusing incompatible resume")
        else:
            write_json(state, cfg)

    def path(self, name):
        return self.root / name

    def get(self, name):
        return read_json(self.path(name))

    def put(self, name, value):
        write_json(self.path(name), value)

    def exists(self, name):
        return self.path(name).exists()

    def signature(self):
        return digest({"config": self.config_hash, "manifest": self.get("manifest.json"),
                       "profile": self.get("profile.json")["selected"]})

    def start_clock(self):
        if not self.exists("clock.json"):
            self.put("clock.json", {"elapsed": 0.0})
        self._started = time.monotonic()

    def checkpoint_clock(self):
        elapsed = self.get("clock.json")["elapsed"] if self.exists("clock.json") else 0.0
        if hasattr(self, "_started"):
            elapsed += time.monotonic() - self._started
            self._started = time.monotonic()
        self.put("clock.json", {"elapsed": elapsed})
        return elapsed

    def budget_available(self, estimate=0):
        return self.checkpoint_clock() + estimate <= self.cfg["budget"]["minutes"] * 60


def draw_mixtures(rng, n, floor=.05):
    return floor + (1 - 3 * floor) * rng.dirichlet(np.ones(3), size=n)


def analytical_policy(theta, floor=.05):
    positive = np.maximum(np.asarray(theta, dtype=float), 0)
    raw = positive / positive.sum() if positive.sum() > 0 else np.ones(3) / 3
    return floor + (1 - 3 * floor) * raw


def progress(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)
