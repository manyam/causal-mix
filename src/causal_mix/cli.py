from __future__ import annotations

import argparse
import fcntl
import os
import platform
import subprocess
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

from .common import BudgetStop, Store, load_config, progress


def environment(store):
    packages = ["torch", "transformers", "peft", "datasets", "huggingface-hub", "econml", "lightgbm", "numpy", "scikit-learn"]
    current = {"python": platform.python_version(), "platform": platform.platform(),
               "packages": {p: version(p) for p in packages}}
    if store.exists("environment.json"):
        previous = store.get("environment.json")
        if previous["packages"] != current["packages"] or previous["python"] != current["python"]:
            raise ValueError("Runtime versions changed; restore the locked environment or use a new artifact directory")
    else:
        store.put("environment.json", current)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local CausalMix method demonstration")
    parser.add_argument("command", choices=["prepare", "benchmark", "synthetic", "collect", "fit", "compare", "report", "demo"])
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args(argv)
    cache = Path(".cache").resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    root = Path(cfg["artifact_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        progress("Another command is using this artifact directory; wait for it to finish.")
        lock.close()
        return 1
    store = Store(cfg)
    environment(store)
    from .data import prepare
    from .report import report

    active = False
    try:
        if args.command == "prepare":
            prepare(store)
            return 0
        if args.command == "report":
            report(store)
            return 0
        if args.command == "demo":
            prepare(store)
        # All experiment stages are offline. Authentication is required only during preparation.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        active = True
        stages = ("synthetic", "benchmark", "collect", "fit", "compare") if args.command == "demo" else (args.command,)
        # Homebrew LightGBM and PyTorch ship different OpenMP runtimes on macOS.
        # Isolate estimation from training instead of allowing unsafe duplicate runtimes.
        for stage in stages:
            store.put("status.json", {"state": "running", "stage": stage})
            code = subprocess.call([sys.executable, "-m", "causal_mix.worker", stage, str(Path(args.config).resolve())])
            if code:
                if store.get("status.json")["state"] == "running":
                    store.put("status.json", {"state": "failed", "stage": stage, "worker_exit_code": code})
                report(store)
                return code
        if args.command == "demo":
            report(store)
        return 0
    except BudgetStop as exc:
        progress(str(exc))
        store.put("status.json", {"state": "budget_limited", "message": str(exc)})
        report(store)
        return 2
    except KeyboardInterrupt:
        store.put("status.json", {"state": "interrupted", "message": "Resume the same command to retry the unfinished run."})
        report(store)
        return 130
    except Exception as exc:
        # Do not serialize exception text from network preparation: some clients embed authentication details.
        store.put("status.json", {"state": "failed", "stage": args.command, "exception_type": type(exc).__name__})
        if args.command == "prepare" or (args.command == "demo" and not active):
            progress(f"Preparation failed ({type(exc).__name__}); verify network access and HF_TOKEN permissions.")
        else:
            traceback.print_exc()
        report(store)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
