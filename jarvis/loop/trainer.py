from __future__ import annotations
import hashlib, json, shutil, subprocess, sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

GIB = 1024 ** 3
SEQ_LADDER = (4096, 3072, 2048, 1536, 1024, 768, 512)
OOM_EXIT = 3
SCRIPT_SRC = Path(__file__).with_name("train_lora.py")


class TrainError(RuntimeError):
    pass


class DoesNotFit(RuntimeError):
    pass


# ----------------------------------------------------------------------------- sizing
@dataclass
class ModelSpec:
    hidden: int
    layers: int
    ffn: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int = 0
    tied: bool = False

    def __post_init__(self):
        if not self.head_dim:
            self.head_dim = self.hidden // self.heads

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def emb_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tied else 2)

    def body_params(self) -> int:
        h, q, kv, f = self.hidden, self.q_dim, self.kv_dim, self.ffn
        return self.layers * (h * q + 2 * h * kv + q * h + 3 * h * f)

    def n_params(self) -> int:
        return self.emb_params + self.body_params()

    def lora_params(self, r: int) -> int:
        """LoRA on every linear in each block (q,k,v,o,gate,up,down) == peft target_modules='all-linear'."""
        h, q, kv, f = self.hidden, self.q_dim, self.kv_dim, self.ffn
        per = r * (h + q) + 2 * r * (h + kv) + r * (q + h) + 3 * r * (h + f)
        return self.layers * per

    @classmethod
    def from_hf_config(cls, src) -> "ModelSpec":
        c = src if isinstance(src, dict) else json.loads(Path(src).read_text())
        c = c.get("text_config", c)
        heads = c["num_attention_heads"]
        return cls(hidden=c["hidden_size"], layers=c["num_hidden_layers"], ffn=c["intermediate_size"],
                   vocab=c["vocab_size"], heads=heads, kv_heads=c.get("num_key_value_heads", heads),
                   head_dim=c.get("head_dim") or 0, tied=bool(c.get("tie_word_embeddings", False)))


def estimate_vram(spec: ModelSpec, r: int, seq_len: int, micro_batch: int, *, quantized: bool = True,
                  trim_logits: bool = True, answer_cap: int = 256) -> dict:
    """HEURISTIC peak-VRAM estimate in GiB for QLoRA + gradient checkpointing + 8-bit paged AdamW.
    Dominant terms: 4-bit body + 16-bit embeddings/lm_head, fp32 LoRA state, checkpointed activations, fp32 logits.
    Verify with a real probe run; the trainer also retries smaller plans on a real OOM."""
    body_bits = 4.5 if quantized else 16.0              # nf4 + double-quant constants ~4.1-4.5 bits/param
    weights = spec.body_params() * body_bits / 8 + spec.emb_params * 2
    lora_state = spec.lora_params(r) * (4 + 4 + 2)       # fp32 weights + fp32 grads + 8-bit adam states
    tokens = micro_batch * seq_len
    h = spec.hidden
    saved = spec.layers * tokens * h * 2                 # one bf16 hidden state per layer (checkpoints)
    layer_peak = tokens * h * 34 * max(1.0, spec.ffn / (4 * h))   # one layer recomputed (sdpa/flash attention)
    logit_tokens = micro_batch * (min(seq_len, answer_cap + 1) if trim_logits else seq_len)
    logits = logit_tokens * spec.vocab * 4 * 3           # fp32 logits + softmax + grad
    parts = dict(weights=weights, lora_state=lora_state, ckpt_acts=saved, layer_peak=layer_peak, logits=logits)
    subtotal = sum(parts.values())
    parts = {k: v / GIB for k, v in parts.items()}
    parts["overhead"] = subtotal * 0.08 / GIB + 0.7      # allocator fragmentation + CUDA context
    parts["total_gb"] = subtotal * 1.08 / GIB + 0.7
    return parts


@dataclass
class TrainPlan:
    r: int
    alpha: int
    seq_len: int
    micro_batch: int
    grad_accum: int
    est_gb: float

    def shrunk(self, spec: ModelSpec, effective_batch: int, **kw) -> Optional["TrainPlan"]:
        """Next-smaller plan after a real OOM: micro-batch 1 -> shorter seq -> smaller rank."""
        r, seq, mb = self.r, self.seq_len, self.micro_batch
        if mb > 1:
            mb = 1
        else:
            lower = [s for s in SEQ_LADDER if s < seq]
            if lower:
                seq = lower[0]
            elif r > 4:
                r //= 2
            else:
                return None
        est = estimate_vram(spec, r, seq, mb, **kw)["total_gb"]
        return TrainPlan(r, 2 * r, seq, mb, max(1, -(-effective_batch // mb)), est)


def plan_training(spec: ModelSpec, vram_gb: float = 6.0, *, headroom: float = 0.88, effective_batch: int = 16,
                  max_r: int = 16, seq_cap: int = 2048, quantized: bool = True, trim_logits: bool = True,
                  answer_cap: int = 256) -> TrainPlan:
    budget = vram_gb * headroom
    kw = dict(quantized=quantized, trim_logits=trim_logits, answer_cap=answer_cap)
    for seq in (s for s in SEQ_LADDER if s <= seq_cap):
        for r in sorted({max_r, max(4, max_r // 2)}, reverse=True):
            for mb in (4, 2, 1):
                est = estimate_vram(spec, r, seq, mb, **kw)["total_gb"]
                if est <= budget:
                    return TrainPlan(r, 2 * r, seq, mb, max(1, -(-effective_batch // mb)), est)
    floor = estimate_vram(spec, max(4, max_r // 2), SEQ_LADDER[-1], 1, **kw)["total_gb"]
    raise DoesNotFit(f"{spec.n_params() / 1e9:.2f}B params: smallest plan needs ~{floor:.1f} GiB > budget {budget:.1f} GiB")


# ----------------------------------------------------------------------------- trainers
@dataclass
class TrainResult:
    artifact: str
    metrics: dict = field(default_factory=dict)


class DryRunTrainer:
    """No-GPU stand-in: writes a deterministic stub 'adapter'. For CI, demos and dry loop runs."""

    def train(self, dataset_dir, out_dir) -> TrainResult:
        ds = Path(dataset_dir) / "train.jsonl"
        data = ds.read_bytes()
        adapter = Path(out_dir) / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(hashlib.sha256(data).digest())
        (adapter / "adapter_config.json").write_text(json.dumps({"dry_run": True}))
        return TrainResult(str(adapter), {"dry_run": True, "examples": data.count(b"\n")})


class CommandTrainer:
    """Any external trainer: argv template with {dataset} (train.jsonl) and {out}. Artifact = {out}/adapter if present."""

    def __init__(self, argv: list, timeout: float = 6 * 3600, runner: Callable = subprocess.run):
        self.argv, self.timeout, self.runner = argv, timeout, runner

    def train(self, dataset_dir, out_dir) -> TrainResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        argv = [a.format(dataset=str(Path(dataset_dir) / "train.jsonl"), out=str(out)) for a in self.argv]
        p = self.runner(argv, capture_output=True, text=True, timeout=self.timeout)
        (out / "train.log").write_text((p.stdout or "") + (p.stderr or ""))
        if p.returncode != 0:
            raise TrainError(f"trainer exit {p.returncode}: {(p.stderr or '')[-300:]}")
        art = out / "adapter"
        return TrainResult(str(art if art.exists() else out), {})


class LoraTrainer:
    """QLoRA SFT on the curated traces via the bundled standalone train_lora.py (run in a subprocess).
    Always trains from the BASE model on the full curated set: reproducible, no compounding drift.
    On a real CUDA OOM (exit 3) it retries with the next-smaller plan."""

    def __init__(self, base_model: str, spec: ModelSpec, *, vram_gb: float = 6.0, epochs: int = 2, lr: float = 2e-4,
                 effective_batch: int = 16, quantize: bool = True, trim_logits: bool = True, answer_cap: int = 256,
                 seq_cap: int = 2048, max_attempts: int = 4, timeout_s: float = 6 * 3600, python: str = sys.executable,
                 runner: Callable = subprocess.run, extra: Optional[dict] = None):
        self.base_model, self.spec = base_model, spec
        self.vram_gb, self.epochs, self.lr, self.eb = vram_gb, epochs, lr, effective_batch
        self.quantize, self.trim, self.answer_cap, self.seq_cap = quantize, trim_logits, answer_cap, seq_cap
        self.max_attempts, self.timeout_s, self.python, self.runner = max_attempts, timeout_s, python, runner
        self.extra = extra or {}

    def _kw(self) -> dict:
        return dict(quantized=self.quantize, trim_logits=self.trim, answer_cap=self.answer_cap)

    def train(self, dataset_dir, out_dir) -> TrainResult:
        dataset_dir, out = Path(dataset_dir), Path(out_dir)
        ds = dataset_dir / "train.jsonl"
        if not ds.exists() or ds.stat().st_size == 0:
            raise TrainError("empty training set")
        out.mkdir(parents=True, exist_ok=True)
        script = out / "train_lora.py"
        shutil.copy(SCRIPT_SRC, script)
        plan = plan_training(self.spec, self.vram_gb, effective_batch=self.eb, seq_cap=self.seq_cap,
                             quantized=self.quantize, trim_logits=self.trim, answer_cap=self.answer_cap)
        log = out / "train.log"
        log.write_text("")
        for attempt in range(1, self.max_attempts + 1):
            cfg = {"base_model": self.base_model, "dataset": str(ds), "out": str(out / "adapter"),
                   "metrics": str(out / "metrics.json"), "epochs": self.epochs, "lr": self.lr,
                   "quantize": self.quantize, "trim_logits": self.trim, "answer_cap": self.answer_cap,
                   "r": plan.r, "alpha": plan.alpha, "seq_len": plan.seq_len, "micro_batch": plan.micro_batch,
                   "grad_accum": plan.grad_accum, "seed": 0, **self.extra}
            cpath = out / "config.json"
            cpath.write_text(json.dumps(cfg, indent=2))
            p = self.runner([self.python, str(script), "--config", str(cpath)], capture_output=True, text=True,
                            timeout=self.timeout_s)
            with open(log, "a") as f:
                f.write(f"\n=== attempt {attempt} plan={asdict(plan)} rc={p.returncode} ===\n")
                f.write((p.stdout or "") + (p.stderr or ""))
            if p.returncode == 0:
                mpath = out / "metrics.json"
                m = json.loads(mpath.read_text()) if mpath.exists() else {}
                sk, ex = sum((m.get("skipped") or {}).values()), m.get("examples", 0)
                if sk and sk / (sk + ex) > 0.2:
                    m["warning"] = (f"{sk}/{sk + ex} examples skipped (too long / answer too long): raise seq_cap or "
                                    f"answer_cap, or trim recalled context")
                return TrainResult(str(out / "adapter"), {**m, "plan": asdict(plan), "attempts": attempt})
            if p.returncode == OOM_EXIT:
                nxt = plan.shrunk(self.spec, self.eb, **self._kw())
                if nxt is None:
                    break
                plan = nxt
                continue
            raise TrainError(f"train_lora.py exit {p.returncode}: {(p.stderr or p.stdout or '')[-400:]}")
        raise TrainError("CUDA out of memory at every plan")
