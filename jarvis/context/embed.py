"""Embedders. `hash` needs no model: it hashes words, bigrams and char-trigrams, so it is lexical with only mild fuzz
and no semantics. `openai` talks to any OpenAI-compatible /v1/embeddings server (llama.cpp, Ollama, vLLM) and is what
gives paraphrase/typo robustness. Both return L2-normalized float32 rows."""
from __future__ import annotations

import json
import math
import re
import unicodedata
import urllib.request
import zlib
from typing import Any, Iterator, Sequence

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


class HashEmbedder:
    name = "hash-v1"
    min_score = 0.18

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _features(text: str) -> Iterator[tuple[str, float]]:
        words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
        for w in words:
            yield "w:" + w, 1.0
        for a, b in zip(words, words[1:]):
            yield f"b:{a} {b}", 0.6
        for w in words:
            if len(w) >= 4:
                p = f"^{w}$"
                grams = [p[i:i + 3] for i in range(len(p) - 2)]
                wt = 0.5 / math.sqrt(len(grams))
                for g in grams:
                    yield "c:" + g, wt

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f, w in self._features(t):
                h = zlib.crc32(f.encode("utf-8"))
                out[i, h % self.dim] += w if (h >> 16) & 1 else -w
        return _normalize(out)


class OpenAIEmbedder:
    min_score = 0.35

    def __init__(self, base_url: str, model: str, timeout: float = 30.0, api_key: str | None = None, batch: int = 32):
        self.base_url, self.model, self.timeout, self.api_key, self.batch = base_url.rstrip("/"), model, timeout, api_key, batch

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for i in range(0, len(texts), self.batch):
            body = json.dumps({"model": self.model, "input": list(texts[i:i + self.batch])}).encode()
            req = urllib.request.Request(self.base_url + "/embeddings", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            rows.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0)))
        return _normalize(np.asarray(rows, dtype=np.float32))


def make_embedder(cfg: dict[str, Any]) -> HashEmbedder | OpenAIEmbedder:
    e = cfg["embed"]
    emb: HashEmbedder | OpenAIEmbedder
    if e.get("backend") == "openai":
        emb = OpenAIEmbedder(e["base_url"], e["model"], float(e.get("timeout_s", 30)))
    else:
        emb = HashEmbedder(int(e.get("dim", 384)))
    if e.get("min_score") is not None:
        emb.min_score = float(e["min_score"])
    return emb
