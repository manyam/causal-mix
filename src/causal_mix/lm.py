from __future__ import annotations

import gc
import time

import numpy as np
import torch

from .common import DOMAINS


def encode_example(tokenizer, row, length):
    prefix = f"Instruction:\n{row['prompt']}\n\nResponse:\n"
    prompt = tokenizer.encode(prefix, add_special_tokens=False)
    response = tokenizer.encode(row["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Leave a context token so causal shifting never discards the first response label.
    kept_response = response[:length - 1]
    prompt_budget = length - len(kept_response)
    # Retain the response delimiter, even when long instructions must be truncated.
    if len(prompt) > prompt_budget:
        tail = min(8, prompt_budget)
        kept_prompt = prompt[:prompt_budget - tail] + prompt[-tail:]
    else:
        kept_prompt = prompt
    ids = kept_prompt + kept_response
    return {"input_ids": ids, "labels": [-100] * len(kept_prompt) + kept_response,
            "prompt_truncated": len(kept_prompt) < len(prompt),
            "response_truncated": len(kept_response) < len(response)}


def collate(rows, pad_id, device):
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


class LanguageModel:
    def __init__(self, store, role, length):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.store, self.role, self.length = store, role, length
        self.device = store.cfg["device"]
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable. Select device: cpu in a separate configuration; rebenchmark.")
        torch.set_num_threads(store.cfg["training"]["threads"])
        torch.manual_seed(store.cfg["seed"])
        path = store.path(f"models/{role}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                   torch_dtype=torch.float32, attn_implementation="sdpa")
        base.config.use_cache = False
        tc = store.cfg["training"]
        self.model = get_peft_model(base, LoraConfig(r=tc["rank"], lora_alpha=tc["alpha"],
                                    lora_dropout=0, target_modules=tc["target_modules"],
                                    bias="none", task_type="CAUSAL_LM"))
        self.initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model).items()}
        self.model.to(self.device)
        self.encoded = {}
        self.reset()

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def reset(self):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, self.initial)
        torch.manual_seed(self.store.cfg["seed"])
        self.model.zero_grad(set_to_none=True)

    def encode(self, row):
        if row["id"] not in self.encoded:
            self.encoded[row["id"]] = encode_example(self.tokenizer, row, self.length)
        return self.encoded[row["id"]]

    def evaluate(self, domains):
        self.model.eval()
        result = {}
        batch_size = self.store.cfg["training"]["batch_size"]
        with torch.no_grad():
            for domain, rows in domains.items():
                losses = []
                for start in range(0, len(rows), batch_size):
                    batch = collate([self.encode(r) for r in rows[start:start + batch_size]],
                                    self.tokenizer.pad_token_id, self.device)
                    labels = batch.pop("labels")[:, 1:]
                    logits = self.model(**batch).logits[:, :-1, :].contiguous()
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                labels.reshape(-1), reduction="none", ignore_index=-100).reshape(labels.shape)
                    counts = (labels != -100).sum(1)
                    if (counts == 0).any():
                        raise ValueError("An evaluation example has no supervised response tokens")
                    losses.extend((loss.sum(1) / counts).cpu().tolist())
                result[domain] = losses
        self.sync()
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Non-finite evaluation loss")
        return result

    def train(self, rows, mixture, steps, seed):
        self.reset()
        self.model.train()
        tc = self.store.cfg["training"]
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        rng = np.random.default_rng(seed)
        counts, token_counts, losses = np.zeros(3, dtype=int), np.zeros(3, dtype=int), []
        self.sync()
        start = time.monotonic()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.
            for _ in range(tc["accumulation_steps"]):
                examples = []
                for domain_index in rng.choice(3, size=tc["batch_size"], p=mixture):
                    domain = DOMAINS[domain_index]
                    row = rows[domain][rng.integers(len(rows[domain]))]
                    encoded = self.encode(row)
                    examples.append(encoded)
                    counts[domain_index] += 1
                    token_counts[domain_index] += sum(t != -100 for t in encoded["labels"])
                batch = collate(examples, self.tokenizer.pad_token_id, self.device)
                loss = self.model(**batch).loss / tc["accumulation_steps"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optimizer.step()
            losses.append(step_loss)
        self.sync()
        elapsed = time.monotonic() - start
        del optimizer
        return {"seconds": elapsed, "loss": losses, "domain_counts": counts.tolist(),
                "realized_mixture": (counts / counts.sum()).tolist(),
                "response_token_counts": token_counts.tolist()}

    def truncation(self):
        rows = list(self.encoded.values())
        return {"examples": len(rows), "prompt_truncated": sum(r["prompt_truncated"] for r in rows),
                "response_truncated": sum(r["response_truncated"] for r in rows)}

    def save(self, path):
        self.model.save_pretrained(path)

    def close(self):
        del self.model
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


def mean_losses(losses):
    return {d: float(np.mean(losses[d])) for d in DOMAINS}


def improvement(baseline, losses):
    return float(np.mean([np.mean(baseline[d]) - np.mean(losses[d]) for d in DOMAINS]))
