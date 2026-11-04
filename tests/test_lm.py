from types import SimpleNamespace

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM

from causal_mix.common import DOMAINS
from causal_mix.lm import LanguageModel, collate, encode_example


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 24 for c in text]


def test_assistant_mask_and_truncation():
    tokenizer = Tokenizer()
    r = encode_example(tokenizer, {"prompt": "long " * 100, "answer": "answer"}, 32)
    assert len(r["input_ids"]) == 32
    assert r["prompt_truncated"] and not r["response_truncated"]
    assert r["labels"][0] == -100
    assert [v for v in r["labels"] if v != -100] == tokenizer.encode("answer") + [2]
    long = encode_example(tokenizer, {"prompt": "p", "answer": "a" * 100}, 16)
    assert long["labels"][0] == -100 and sum(v != -100 for v in long["labels"]) == 15
    short = encode_example(tokenizer, {"prompt": "p", "answer": "a"}, 100)
    b = collate([r, short], 2, "cpu")
    assert torch.all(b["labels"][b["attention_mask"] == 0] == -100)


def test_fresh_adapter_and_optimizer_reset(cfg):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    lm = LanguageModel.__new__(LanguageModel)
    lm.model, lm.device, lm.tokenizer, lm.length, lm.encoded = model, "cpu", Tokenizer(), 24, {}
    lm.store = SimpleNamespace(cfg=cfg)
    lm.initial = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    frozen = {k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad}
    rows = {d: [{"id": d, "prompt": "p", "answer": "a"}] for d in DOMAINS}
    a = lm.train(rows, [1 / 3] * 3, 2, 42)
    trained = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    assert any(not torch.equal(v, lm.initial[k]) for k, v in trained.items())
    lm.reset()
    assert all(torch.equal(v, lm.initial[k]) for k, v in get_peft_model_state_dict(model).items())
    b = lm.train(rows, [1 / 3] * 3, 2, 42)
    assert np.allclose(a["loss"], b["loss"])
    assert all(torch.equal(v, trained[k]) for k, v in get_peft_model_state_dict(model).items())
    assert all(torch.equal(v, frozen[k]) for k, v in model.named_parameters() if k in frozen)
