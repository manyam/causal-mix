from __future__ import annotations

import os
import re
import unicodedata

import numpy as np

from .common import DOMAINS, digest, progress


def load_data(store, split):
    rows = store.get(f"data/{split}.json")
    if digest(rows) != store.get("manifest.json")["data_hashes"][split]:
        raise ValueError(f"Prepared {split} data changed; refusing incompatible resume")
    return rows


def normalize_prompt(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def adapt_row(domain, row, source_split):
    if domain == "math":
        prompt, answer = row["question"], row["answer"]
        source_id = digest(row)
    elif domain == "coding":
        prompt, answer = row["text"], row["code"]
        source_id = str(row["task_id"])
    else:
        messages = row["messages"]
        first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
        if first_user is None or first_user + 1 >= len(messages):
            return None
        if messages[first_user + 1]["role"] != "assistant":
            return None
        prompt, answer = messages[first_user]["content"], messages[first_user + 1]["content"]
        source_id = str(row["prompt_id"])
    if not prompt.strip() or not answer.strip():
        return None
    return {"id": f"{domain}:{source_split}:{source_id}", "source_id": source_id,
            "source_split": source_split, "domain": domain, "prompt": prompt,
            "answer": answer, "prompt_hash": digest(normalize_prompt(prompt))}


def select_splits(raw, sizes, seed):
    """Reserve every official test prompt before selecting any training example."""
    reserved = {r["prompt_hash"] for domain in DOMAINS for r in raw[domain]["test"]}
    used = set()
    result = {s: {d: [] for d in DOMAINS} for s in ("train", "dev", "test")}
    for domain in DOMAINS:
        for source in ("test", "train"):
            candidates = sorted(raw[domain][source], key=lambda r: digest([seed, r["id"]]))
            target_splits = ["test"] if source == "test" else ["dev", "train"]
            cursor = 0
            for split in target_splits:
                while len(result[split][domain]) < sizes[split] and cursor < len(candidates):
                    row = candidates[cursor]
                    cursor += 1
                    key = row["prompt_hash"]
                    if key in used or (source == "train" and key in reserved):
                        continue
                    used.add(key)
                    result[split][domain].append(row)
                if len(result[split][domain]) != sizes[split]:
                    raise ValueError(f"Insufficient unique {domain}/{split} examples after deduplication")
    return result


def prepare(store):
    if store.exists("manifest.json"):
        for role in ("proxy", "transfer"):
            if not store.path(f"models/{role}/config.json").exists():
                raise ValueError("Prepared model files are missing; restore artifacts before resuming")
        progress("Prepared assets already present")
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be set for preparation (its value is never logged)")
    api = HfApi(token=token)
    manifest = {"models": {}, "datasets": {}, "schema_version": 1}
    for role, spec in store.cfg["models"].items():
        progress(f"Resolving and downloading {spec['id']}")
        info = api.model_info(spec["id"], revision=spec["revision"])
        snapshot_download(spec["id"], revision=info.sha, token=token,
                          local_dir=store.path(f"models/{role}"),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "README.md"],
                          max_workers=4)
        manifest["models"][role] = {"id": spec["id"], "revision": info.sha}
    raw = {}
    for domain in DOMAINS:
        spec = store.cfg["datasets"][domain]
        progress(f"Preparing {spec['id']} / {spec['config']}")
        info = api.dataset_info(spec["id"], revision=spec["revision"])
        ds = load_dataset(spec["id"], spec["config"], revision=info.sha, token=token,
                          cache_dir=str(store.path("hub_datasets")))
        raw[domain] = {}
        for split in ("train", "test"):
            raw[domain][split] = [r for row in ds[split] if (r := adapt_row(domain, row, split))]
        card = info.card_data.to_dict() if info.card_data else {}
        manifest["datasets"][domain] = {"id": spec["id"], "revision": info.sha,
                                        "config": spec["config"], "license": card.get("license")}
    selected = select_splits(raw, store.cfg["data"], store.cfg["seed"])
    for split, rows in selected.items():
        store.put(f"data/{split}.json", rows)
    manifest["sample_ids"] = {s: {d: [r["id"] for r in rows[d]] for d in DOMAINS}
                              for s, rows in selected.items()}
    manifest["data_hashes"] = {s: digest(rows) for s, rows in selected.items()}
    store.put("manifest.json", manifest)
    progress("Preparation complete; experiment stages use local assets only")


def make_pools(rows, losses, config, count):
    """Pool RNG is separate from mixture RNG; features depend on pool membership only."""
    pools = []
    for pool_id in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 100, pool_id]))
        members, features, difficulties = {}, [], {}
        for domain in DOMAINS:
            nll = np.asarray(losses[domain])
            difficulty = rng.uniform(*config["data"]["difficulty_range"])
            logits = difficulty * (nll - nll.mean()) / max(nll.std(), 1e-6)
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            indices = rng.choice(len(nll), config["data"]["pool_per_domain"], replace=False, p=weights)
            selected = [rows[domain][i] for i in indices]
            members[domain] = [r["id"] for r in selected]
            features.extend(domain_features(selected, nll[indices]))
            difficulties[domain] = float(difficulty)
        pools.append({"pool_id": pool_id, "members": members, "x": features,
                      "difficulty": difficulties})
    return pools


def domain_features(rows, losses):
    return [float(np.mean(losses)), float(np.mean([len(r["answer"]) for r in rows])),
            float(np.mean([len(set(r["answer"])) / len(r["answer"]) for r in rows]))]
