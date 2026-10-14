"""VRAM budget estimator for the 6 GB RTX 3050. Estimates, +/-15%: verify against the llama.cpp
load log (`llm_load_tensors: ... buffer size`, `KV self size`) and `nvidia-smi`.
Architecture constants are from the public model configs as I know them; check GGUF metadata
(`gguf-dump model.gguf | grep -E 'block_count|head_count_kv|embedding_length'`)."""
from __future__ import annotations

MODELS = {  # params (B), layers, kv_heads, head_dim
    "qwen2.5-3b": (3.09, 36, 2, 128),
    "qwen3-4b": (4.02, 36, 8, 128),
    "qwen2.5-7b": (7.62, 28, 4, 128),
    "llama3.2-3b": (3.21, 28, 8, 128),
}
QUANT_BPW = {"q4_0": 4.55, "q4_k_m": 4.85, "q5_k_m": 5.69, "q6_k": 6.56, "q8_0": 8.5}
KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
AUX_MB = {"stt-gpu (distil-small int8)": 450, "stt-gpu (small int8)": 600, "kokoro-gpu": 700}
RUNTIME_MB = 350  # CUDA context + llama.cpp compute buffers (grows slowly with ctx/batch)
MB = 1024 * 1024


def llm_mb(model: str, quant: str, ctx: int, kv: str = "q8_0") -> dict[str, float]:
    p, layers, kvh, hd = MODELS[model]
    weights = p * 1e9 * QUANT_BPW[quant] / 8 / MB
    kv_cache = 2 * layers * kvh * hd * ctx * KV_BYTES[kv] / MB
    return {"weights": weights, "kv_cache": kv_cache, "runtime": RUNTIME_MB,
            "total": weights + kv_cache + RUNTIME_MB}


def plan(total_mb: int = 6144, display_reserve_mb: int = 0, extras: tuple[str, ...] = ()) -> list[dict]:
    """Every model/quant/ctx combo with headroom after extras. Sorted best-fit first."""
    aux = sum(AUX_MB[e] for e in extras)
    rows = []
    for m in MODELS:
        for q in ("q4_k_m", "q5_k_m", "q6_k", "q8_0"):
            for ctx in (2048, 4096, 8192):
                b = llm_mb(m, q, ctx)
                free = total_mb - display_reserve_mb - aux - b["total"]
                rows.append({"model": m, "quant": q, "ctx": ctx, "llm_mb": round(b["total"]),
                             "aux_mb": aux, "free_mb": round(free), "fits": free >= 512})
    rows.sort(key=lambda r: (not r["fits"], -MODELS[r["model"]][0], -QUANT_BPW[r["quant"]], -r["ctx"]))
    return rows


def format_plan(rows: list[dict], only_fits: bool = True, limit: int = 12) -> str:
    out = [f"{'model':<13}{'quant':<8}{'ctx':>6}{'llm MB':>8}{'aux MB':>8}{'free MB':>9}  fits"]
    for r in [r for r in rows if r["fits"] or not only_fits][:limit]:
        out.append(f"{r['model']:<13}{r['quant']:<8}{r['ctx']:>6}{r['llm_mb']:>8}{r['aux_mb']:>8}{r['free_mb']:>9}  {'yes' if r['fits'] else 'NO'}")
    return "\n".join(out)
