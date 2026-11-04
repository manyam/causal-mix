# Completed local demonstration

This read-only snapshot contains the 2026-11-04 run's numerical records, report, chart, source revisions, sample identifiers, and ten final LoRA adapters. It excludes raw dataset text, downloaded base models, caches, credentials, and debugging logs.

Open `report.html` locally with `results.png` beside it. The root project README explains the experiment and signal analysis.

Comparison `adapter_path` values are relative to this snapshot. Adapter configurations use the original Hugging Face model IDs and pinned revisions instead of machine-local paths; weight files are unchanged. Historical configuration hashes and run signatures refer to the original experiment. This export is for inspection and adapter reuse, not in-place training resume.

Load an adapter after downloading its pinned base model:

```python
import os
from pathlib import Path
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = Path(__file__).parent / 'adapters/proxy/causalmix_a'
config = PeftConfig.from_pretrained(adapter)
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, revision=config.revision, token=os.environ['HF_TOKEN'])
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, revision=config.revision, token=os.environ['HF_TOKEN'])
model = PeftModel.from_pretrained(base, adapter)
```

The example assumes a Python script saved beside this README. The adapters use the plain instruction/response format documented in the project README.

`checksums.sha256` records every exported file except itself. Verify with `shasum -a 256 -c checksums.sha256` from this directory.
