import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))


@pytest.fixture
def cfg(tmp_path):
    from causal_mix.common import load_config
    c = load_config("configs/demo.yaml")
    c["artifact_dir"] = str(tmp_path / "experiment")
    return c
