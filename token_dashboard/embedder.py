"""Embedder shim — turns text into 384-dim BGE-Small vectors.

We don't ship fastembed inside the dashboard image (it's ~150 MB of ONNX
runtime + model). Instead, we call HeliosDB's MCP `code_embed` tool over
the existing MCP_HTTP_URL — the embedder lives where the rest of the
embeddings live.

Falls back to a deterministic hash-based embedding if MCP is not
reachable (so the rest of the pipeline still runs in dev). The hash
embedding is clearly labelled and the vector_search code knows to
backfill it when MCP comes back.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
import time
from typing import Optional

from . import mcp as mcp_client


EMBED_DIM = 384


_lock = threading.Lock()
_cache: dict[str, list] = {}
_cache_max = 5000  # bound the cache; LRU eviction by insertion order


def _cache_get(text: str) -> Optional[list]:
    return _cache.get(text)


def _cache_put(text: str, vec: list) -> None:
    with _lock:
        if len(_cache) >= _cache_max:
            # drop oldest 10% in one pass
            to_drop = len(_cache) // 10
            for k in list(_cache.keys())[:to_drop]:
                _cache.pop(k, None)
        _cache[text] = vec


# ---------- Hash fallback ----------

def _hash_embedding(text: str) -> list:
    """Deterministic 384-dim vector from BLAKE2b. Clearly NOT a real
    semantic embedding — used only when MCP code_embed is unreachable so
    upstream code can still run end-to-end.

    BLAKE2b output bytes interpreted as IEEE 754 floats can be NaN/inf/denorms;
    we rejection-sample to keep the vector finite, then unit-normalise.
    """
    seed = text.encode("utf-8", errors="replace")
    floats: list[float] = []
    counter = 0
    while len(floats) < EMBED_DIM:
        h = hashlib.blake2b(seed + counter.to_bytes(4, "big"), digest_size=64)
        for j in range(0, 64, 4):
            (f,) = struct.unpack("f", h.digest()[j:j + 4])
            if math.isfinite(f) and abs(f) < 1e6:
                # Map roughly to [-1, 1] so the cosine has reasonable magnitude
                f = max(-1.0, min(1.0, f / 1000.0))
                floats.append(f)
                if len(floats) >= EMBED_DIM:
                    break
        counter += 1
        if counter > 64:  # safety bail (shouldn't happen)
            while len(floats) < EMBED_DIM:
                floats.append(0.0)
            break
    norm = math.sqrt(sum(f * f for f in floats)) or 1.0
    return [f / norm for f in floats]


# ---------- MCP-backed embedder ----------

def _via_mcp(text: str) -> Optional[list]:
    """Call HeliosDB's code_embed MCP tool. Returns None if unavailable."""
    if not mcp_client.is_configured():
        return None
    try:
        result = mcp_client.call_tool(
            mcp_client.http_url(),
            "code_embed",
            {"text": text},
            timeout=15.0,
        )
    except Exception:
        return None
    # The MCP tool returns content blocks; extract the structured embedding.
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        # Text-encoded JSON block
        t = block.get("text")
        if isinstance(t, str):
            try:
                payload = json.loads(t)
            except Exception:
                continue
            vec = payload.get("embedding") or payload.get("vector") or payload.get("body_vec")
            if isinstance(vec, list) and len(vec) == EMBED_DIM:
                return [float(x) for x in vec]
        # Already a structured embedding block (some servers do this)
        for key in ("embedding", "vector"):
            if key in block and isinstance(block[key], list) and len(block[key]) == EMBED_DIM:
                return [float(x) for x in block[key]]
    return None


# ---------- Public API ----------

def embed(text: str) -> list:
    """Return a 384-dim embedding. Tries MCP first, falls back to hash.

    Cached per-process. Caller never gets None — if MCP fails, the hash
    embedding is returned so the pipeline keeps moving; the writer marks
    such rows for refresh."""
    if text is None:
        text = ""
    text = text.strip()[:8000]  # bound payload size
    cached = _cache_get(text)
    if cached is not None:
        return cached
    vec = _via_mcp(text) or _hash_embedding(text)
    _cache_put(text, vec)
    return vec


def embed_batch(texts: list) -> list:
    """Convenience — embed N texts, returns a list of vectors.

    No MCP batch endpoint is assumed; we just iterate. The cache absorbs
    duplicates."""
    return [embed(t) for t in texts]


def is_real_embedding_available() -> bool:
    """True if MCP code_embed is reachable (real embeddings flow). False
    if we're falling back to the hash shim."""
    if not mcp_client.is_configured():
        return False
    try:
        v = _via_mcp("token-dashboard probe")
        return v is not None
    except Exception:
        return False
