"""Export a portable, read-only result snapshot without dataset text or base weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def package(source: Path, destination: Path):
    if destination.exists():
        raise ValueError("Destination exists; select a new snapshot directory")
    status = json.loads((source / "status.json").read_text())
    if status["state"] != "complete":
        raise ValueError("Only completed experiments can be packaged")
    manifest = json.loads((source / "manifest.json").read_text())
    destination.mkdir(parents=True)
    files = ["config.json", "environment.json", "manifest.json", "synthetic.json",
             "benchmark.json", "profile.json", "pools.json", "training_losses.json",
             "policies.json", "status.json", "clock.json", "report.html", "results.png"]
    for name in files:
        shutil.copy2(source / name, destination / name)
    for folder in ("runs", "baselines", "comparisons"):
        target = destination / folder
        target.mkdir()
        for path in sorted((source / folder).glob("*.json")):
            record = json.loads(path.read_text())
            if folder == "comparisons":
                record["adapter_path"] = f"adapters/{record['role']}/{record['policy']}"
            dump(target / path.name, record)
    adapter_count = 0
    for role, model in manifest["models"].items():
        for folder in sorted((source / "adapters" / role).iterdir()):
            if not folder.is_dir():
                continue
            target = destination / "adapters" / role / folder.name
            target.mkdir(parents=True)
            config = json.loads((folder / "adapter_config.json").read_text())
            config["base_model_name_or_path"] = model["id"]
            config["revision"] = model["revision"]
            dump(target / "adapter_config.json", config)
            shutil.copy2(folder / "adapter_model.safetensors", target / "adapter_model.safetensors")
            (target / "README.md").write_text(
                f"# {role.title()} adapter: {folder.name}\n\n"
                f"Base model: [{model['id']}](https://huggingface.co/{model['id']}); "
                f"revision `{model['revision']}`.\n\n"
                "Toy CausalMix demonstration: rank-8 LoRA on q/v projections, 16 optimizer steps, "
                "256-token sequences, seed 42. These weights are the original final-run adapter weights.\n\n"
                "See the snapshot README, configuration, manifest, and comparison records for data provenance "
                "and measured results. Training datasets: GSM8K (MIT), MBPP (CC-BY-4.0), and "
                "No Robots (CC-BY-NC-4.0). This is a research demonstration, not an evaluated chat assistant.\n")
            adapter_count += 1
    if adapter_count != status["comparison_runs"]:
        raise ValueError("Adapter count does not match completed comparisons")
    (destination / "README.md").write_text(
        "# Completed local demonstration\n\n"
        "This read-only snapshot contains the 2026-11-04 run's numerical records, report, chart, "
        "source revisions, sample identifiers, and ten final LoRA adapters. It excludes raw dataset "
        "text, downloaded base models, caches, credentials, and debugging logs.\n\n"
        "Open `report.html` locally with `results.png` beside it. The root project README explains "
        "the experiment and signal analysis.\n\n"
        "Comparison `adapter_path` values are relative to this snapshot. Adapter configurations "
        "use the original Hugging Face model IDs and pinned revisions instead of machine-local paths; "
        "weight files are unchanged. Historical configuration hashes and run signatures refer to the "
        "original experiment. This export is for inspection and adapter reuse, not in-place training resume.\n\n"
        "Load an adapter after downloading its pinned base model:\n\n"
        "```python\n"
        "import os\n"
        "from pathlib import Path\n"
        "from peft import PeftConfig, PeftModel\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        "adapter = Path(__file__).parent / 'adapters/proxy/causalmix_a'\n"
        "config = PeftConfig.from_pretrained(adapter)\n"
        "base = AutoModelForCausalLM.from_pretrained(\n"
        "    config.base_model_name_or_path, revision=config.revision, token=os.environ['HF_TOKEN'])\n"
        "tokenizer = AutoTokenizer.from_pretrained(\n"
        "    config.base_model_name_or_path, revision=config.revision, token=os.environ['HF_TOKEN'])\n"
        "model = PeftModel.from_pretrained(base, adapter)\n"
        "```\n\n"
        "The example assumes a Python script saved beside this README. The adapters use the plain "
        "instruction/response format documented in the project README.\n\n"
        "`checksums.sha256` records every exported file except itself. Verify with "
        "`shasum -a 256 -c checksums.sha256` from this directory.\n")
    entries = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(destination).as_posix()}")
    (destination / "checksums.sha256").write_text("\n".join(entries) + "\n")
    print(f"Packaged {len(entries)} files and {adapter_count} adapters in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--output", type=Path, default=Path("results/demo-2026-11-04"))
    arguments = parser.parse_args()
    package(arguments.source, arguments.output)
