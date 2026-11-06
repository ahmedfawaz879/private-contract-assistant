"""Chunking and per-contract retrieval: BM25, optionally fused with dense embeddings by reciprocal rank."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOK = re.compile(r"[a-z0-9]+")


def toks(s: str) -> list[str]:
    return TOK.findall(s.lower())


def chunk(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of ~``size`` chars, cut at a space in the second half, overlapping by ``overlap``."""
    out, s = [], 0
    while s < len(text):
        e = min(len(text), s + size)
        if e < len(text):
            sp = text.rfind(" ", s + size // 2, e)
            e = sp if sp > 0 else e
        out.append((s, e))
        s = e - overlap if e < len(text) else len(text)
        if out and s <= out[-1][0]:
            s = out[-1][1]
    return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion."""
    sc: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            sc[i] = sc.get(i, 0) + 1 / (k + pos + 1)
    return sorted(sc, key=lambda i: -sc[i])


class DocIndex:
    """Index over one contract's chunks."""

    def __init__(self, text: str, chunk_chars: int = 1200, chunk_overlap: int = 200, embedder=None):
        self.text = text
        self.spans = chunk(text, chunk_chars, chunk_overlap)
        self.chunks = [text[s:e] for s, e in self.spans]
        self.bm25 = BM25Okapi([toks(c) or ["_"] for c in self.chunks])
        self.emb = embedder.encode(self.chunks, normalize_embeddings=True, batch_size=64) if embedder else None

    def search(self, q: str, k: int, embedder=None) -> list[int]:
        r1 = [int(i) for i in np.argsort(-self.bm25.get_scores(toks(q)), kind="stable")]
        if self.emb is None or embedder is None:
            return r1[:k]
        qv = embedder.encode([q], normalize_embeddings=True)[0]
        r2 = [int(i) for i in np.argsort(-(self.emb @ qv), kind="stable")]
        return rrf([r1, r2])[:k]


def load_embedder(model: str, device: str = "cpu"):
    """Sentence-transformers model, or ``None`` (BM25 only) if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model, device=device)
    except Exception as e:  # noqa: BLE001 - any failure means "fall back to BM25"
        print("dense retrieval off (BM25 only):", type(e).__name__, str(e)[:120])
        return None
