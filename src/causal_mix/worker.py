"""Internal stage process. The parent CLI owns the experiment's exclusive lock."""
import sys
import traceback

from .common import BudgetStop, Store, load_config, progress


def main():
    stage, config = sys.argv[1:]
    store = Store(load_config(config))
    store.start_clock()
    try:
        if stage == "synthetic":
            from .causal import synthetic
            synthetic(store)
        else:
            from . import pipeline
            getattr(pipeline, stage)(store)
        return 0
    except BudgetStop as exc:
        progress(str(exc))
        store.put("status.json", {"state": "budget_limited", "stage": stage, "message": str(exc)})
        return 2
    except KeyboardInterrupt:
        store.put("status.json", {"state": "interrupted", "stage": stage})
        return 130
    except Exception as exc:
        traceback.print_exc()
        store.put("status.json", {"state": "failed", "stage": stage, "exception_type": type(exc).__name__})
        return 1
    finally:
        store.checkpoint_clock()


if __name__ == "__main__":
    sys.exit(main())
