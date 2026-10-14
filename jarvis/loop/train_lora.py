#!/usr/bin/env python3
"""Standalone QLoRA SFT loop sized for a 6GB GPU. Copied into every run dir by LoraTrainer.

Design choices that matter for VRAM:
  * 4-bit NF4 base + LoRA (peft) on all linear layers, gradient checkpointing (non-reentrant)
  * loss only on the assistant answer; when the model supports `logits_to_keep`, lm_head runs ONLY on the
    answer window (answers are short JSON) -> avoids the multi-GB fp32 logits of big-vocab models
  * left-padded batches, explicit position_ids, 8-bit paged AdamW (falls back to torch AdamW)
  * no prepare_model_for_kbit_training (it upcasts embeddings/lm_head to fp32 = +1GB on 1.5B models)
Exit codes: 0 ok | 3 CUDA OOM (caller retries a smaller plan) | 4 no usable examples.
Pure helpers (lr_at, build_example, collate_left, window_k) import without torch and are unit-tested.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import time

OOM_EXIT = 3
NO_DATA_EXIT = 4


# ----------------------------------------------------------------------------- pure helpers
def lr_at(step: int, total: int, warmup: int, base: float, floor: float = 0.1) -> float:
    if total <= 0:
        return base
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


def _ids(x):
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    elif hasattr(x, "input_ids"):
        x = x.input_ids
    if len(x) and isinstance(x[0], (list, tuple)):
        x = x[0]
    return [int(i) for i in x]


def build_example(tok, messages, max_len: int, answer_cap: int):
    """-> ((input_ids, labels), 'ok') or (None, reason). Loss only on the final assistant message."""
    prompt_ids = _ids(tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = _ids(tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True))
    n, lim = 0, min(len(prompt_ids), len(full_ids))
    while n < lim and prompt_ids[n] == full_ids[n]:
        n += 1
    if len(full_ids) > max_len:
        return None, "too_long"            # never truncate: a cut-off JSON target teaches broken output
    n_ans = len(full_ids) - n
    if n_ans <= 0:
        return None, "no_answer"
    if n_ans > answer_cap:
        return None, "answer_too_long"
    return (full_ids, [-100] * n + full_ids[n:]), "ok"


def collate_left(batch, pad_id: int):
    L = max(len(ids) for ids, _ in batch)
    ids_o, lab_o, msk_o = [], [], []
    for ids, lab in batch:
        pad = L - len(ids)
        ids_o.append([pad_id] * pad + list(ids))
        lab_o.append([-100] * pad + list(lab))
        msk_o.append([0] * pad + [1] * len(ids))
    return ids_o, lab_o, msk_o


def window_k(labels_batch, seq_len: int) -> int:
    """Trailing positions whose logits are needed: answer tokens + 1 (position p-1 predicts token p)."""
    k = max(sum(1 for x in lab if x != -100) for lab in labels_batch) + 1
    return min(k, seq_len)


def _write_metrics(cfg, **m):
    path = cfg.get("metrics")
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(m, f, indent=2)


def _load_model(cls, base, kw):
    try:
        return cls.from_pretrained(base, **kw)
    except TypeError as e:                 # transformers renamed torch_dtype -> dtype
        if "torch_dtype" in str(e) and "torch_dtype" in kw:
            kw = dict(kw)
            kw["dtype"] = kw.pop("torch_dtype")
            return cls.from_pretrained(base, **kw)
        raise


# ----------------------------------------------------------------------------- training
def run(cfg: dict) -> int:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        cdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # Turing -> fp16 + GradScaler
    else:
        cdtype = torch.float32

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    examples, skipped = [], {}
    with open(cfg["dataset"], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex, why = build_example(tok, json.loads(line)["messages"], cfg["seq_len"], cfg["answer_cap"])
            if ex is None:
                skipped[why] = skipped.get(why, 0) + 1
            else:
                examples.append(ex)
    print(f"examples={len(examples)} skipped={skipped}", flush=True)
    if not examples:
        _write_metrics(cfg, status="no_examples", skipped=skipped)
        return NO_DATA_EXIT

    kw = {"torch_dtype": cdtype}
    quantize = cfg.get("quantize", True)
    if quantize:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cdtype)
        kw["device_map"] = {"": 0}
    if cfg.get("attn"):
        kw["attn_implementation"] = cfg["attn"]
    model = _load_model(AutoModelForCausalLM, cfg["base_model"], kw)
    if not quantize and dev == "cuda":
        model.to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", "all-linear")))
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=cfg["lr"], weight_decay=0.0)
    except Exception:
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    use_scaler = dev == "cuda" and cdtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    keep_name = None
    if cfg.get("trim_logits", True):
        sig = inspect.signature(model.get_base_model().forward).parameters
        keep_name = next((n for n in ("logits_to_keep", "num_logits_to_keep") if n in sig), None)
    print(f"device={dev} dtype={cdtype} quantize={quantize} logits_window={'on:' + keep_name if keep_name else 'off'}",
          flush=True)

    mb, accum, epochs = cfg["micro_batch"], cfg["grad_accum"], cfg["epochs"]
    steps_per_epoch = math.ceil(len(examples) / (mb * accum))
    total = steps_per_epoch * epochs
    if cfg.get("max_steps"):
        total = min(total, int(cfg["max_steps"]))
    warmup = max(1, int(0.05 * total))
    pad_id = tok.pad_token_id
    losses, step = [], 0
    model.train()
    try:
        for ep in range(epochs):
            order = list(range(len(examples)))
            random.Random(seed + ep).shuffle(order)
            batches = [order[i:i + mb] for i in range(0, len(order), mb)]
            micro = 0
            for bi, idxs in enumerate(batches):
                ids, lab, msk = collate_left([examples[i] for i in idxs], pad_id)
                S = len(ids[0])
                input_ids = torch.tensor(ids, device=dev)
                labels = torch.tensor(lab, device=dev)
                mask = torch.tensor(msk, device=dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                extra = {keep_name: window_k(lab, S)} if keep_name else {}
                with torch.autocast(device_type=dev, dtype=cdtype, enabled=(dev == "cuda")):
                    logits = model(input_ids=input_ids, attention_mask=mask, position_ids=pos, **extra).logits
                K = logits.size(1)
                tgt = labels[:, S - K + 1:]                       # logits at i predict token i+1
                lg = logits[:, :-1, :].float()
                n_tok = (tgt != -100).sum().clamp(min=1)
                loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                       ignore_index=-100, reduction="sum") / n_tok
                scaler.scale(loss / accum).backward()
                losses.append(float(loss.detach()))
                micro += 1
                if micro % accum == 0 or bi == len(batches) - 1:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step, total, warmup, cfg["lr"])
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % cfg.get("log_every", 10) == 0:
                        print(f"step {step}/{total} loss {sum(losses[-accum:]) / len(losses[-accum:]):.4f}", flush=True)
                    if step >= total:
                        break
            if step >= total:
                break
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if dev == "cuda" else 0.0
        _write_metrics(cfg, status="oom", step=step, peak_vram_gb=peak)
        print("CUDA OOM", flush=True)
        return OOM_EXIT

    os.makedirs(cfg["out"], exist_ok=True)
    model.save_pretrained(cfg["out"])
    tok.save_pretrained(cfg["out"])
    head, tail = losses[:20], losses[-20:]
    _write_metrics(
        cfg, status="ok", steps=step, examples=len(examples), skipped=skipped,
        first_loss=sum(head) / len(head), final_loss=sum(tail) / len(tail), seconds=time.time() - t0,
        peak_vram_gb=(torch.cuda.max_memory_allocated() / 1024 ** 3) if dev == "cuda" else 0.0)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("train_lora")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N optimizer steps")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    if a.max_steps:
        cfg["max_steps"] = a.max_steps
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
