"""Item #3 — Semantic clustering of prompts.

Mini-batch k-means in-process over `dashboard.messages.body_vec`. Stores
results in `dashboard.clusters` + `dashboard.cluster_members`.

Why in-process: keeps the dashboard self-contained. HeliosDB's plpgsql
could host this too, but the round-trip cost is small for the size of
data the dashboard sees (~hundreds to low-thousands of unique prompts).
"""
from __future__ import annotations

import math
import os
import random
import time
from typing import Optional

from . import helios_writer as hw


def cluster_prompts(k: int = 8, max_iter: int = 20, project_slug: Optional[str] = None,
                    seed: int = 1337) -> dict:
    """Run k-means over prompt embeddings; persist clusters."""
    conn = hw.get_conn()
    if conn is None:
        return {"ok": False, "error": "helios not configured"}
    cur = conn.cursor()
    where = "type='user' AND body_vec IS NOT NULL"
    args: tuple = ()
    if project_slug:
        where += " AND project_slug = $1"
        args = (project_slug,)
    if args:
        cur.execute(f"""
          SELECT uuid, prompt_text, body_vec
            FROM dashboard.messages
           WHERE {where}
           LIMIT 5000
        """, args)
    else:
        cur.execute(f"""
          SELECT uuid, prompt_text, body_vec
            FROM dashboard.messages
           WHERE {where}
           LIMIT 5000
        """)
    rows = cur.fetchall()
    if len(rows) < k * 2:
        return {"ok": False, "error": f"need ≥{k*2} prompts, found {len(rows)}"}
    uuids = [r[0] for r in rows]
    previews = [(r[1] or "")[:120] for r in rows]
    # Vectors come back as a string-ish or list-ish form depending on driver;
    # parse robustly.
    vectors = [_parse_vector(r[2]) for r in rows]

    rng = random.Random(seed)
    # k-means++ init. If all distances collapse to 0 (e.g. when hash
    # embeddings produce near-identical vectors), fall back to uniform pick.
    centroids = [vectors[rng.randrange(len(vectors))]]
    while len(centroids) < k:
        d2 = []
        for v in vectors:
            best = min(_l2(v, c) for c in centroids)
            d2.append(best * best)
        total = sum(d2)
        if total <= 0:
            centroids.append(vectors[rng.randrange(len(vectors))])
            continue
        pick = rng.random() * total
        acc = 0.0
        chosen = None
        for i, w in enumerate(d2):
            acc += w
            if acc >= pick:
                chosen = i
                break
        # acc >= pick must trigger by the end; if it didn't due to FP drift,
        # take the last index.
        centroids.append(vectors[chosen if chosen is not None else len(vectors)-1])
    # Lloyd's iterations
    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        changed = 0
        for i, v in enumerate(vectors):
            best, best_d = 0, float("inf")
            for j, c in enumerate(centroids):
                d = _cosine(v, c)
                if d < best_d:
                    best, best_d = j, d
            if assignments[i] != best:
                assignments[i] = best
                changed += 1
        # Recompute centroids
        new_c = [[0.0] * len(centroids[0]) for _ in range(k)]
        counts = [0] * k
        for i, v in enumerate(vectors):
            j = assignments[i]
            counts[j] += 1
            for d, x in enumerate(v):
                new_c[j][d] += x
        for j in range(k):
            if counts[j]:
                new_c[j] = [x / counts[j] for x in new_c[j]]
            else:
                new_c[j] = vectors[rng.randrange(len(vectors))]
        centroids = new_c
        if changed == 0:
            break

    # Persist
    cur.execute("DELETE FROM dashboard.cluster_members")
    cur.execute("DELETE FROM dashboard.clusters")
    built = time.time()
    label_map: dict[int, str] = {}
    for j, c in enumerate(centroids):
        # Label by the centroid's nearest prompt's first-3-words
        nearest_idx = min(range(len(vectors)), key=lambda i: _cosine(vectors[i], c) if assignments[i] == j else float("inf"))
        words = [w for w in (previews[nearest_idx] or "").split()[:6] if w.isalnum() or "-" in w][:4]
        label = (" ".join(words) or f"cluster-{j}").strip()[:80]
        label_map[j] = label
        size = sum(1 for a in assignments if a == j)
        cur.execute("""
          INSERT INTO dashboard.clusters (id, label, project_slug, centroid, size, built_at)
          VALUES ($1,$2,$3,CAST($4 AS VECTOR(384)),$5,$6)
        """, (j + 1, label, project_slug, hw._vector_literal(c), size, built))
    for i, j in enumerate(assignments):
        cur.execute("""
          INSERT INTO dashboard.cluster_members (cluster_id, uuid, distance)
          VALUES ($1,$2,$3)
        """, (j + 1, uuids[i], float(_cosine(vectors[i], centroids[j]))))
    hw._commit(conn)
    return {
        "ok": True,
        "k": k,
        "n_prompts": len(vectors),
        "labels": [{"id": j + 1, "label": label_map[j],
                    "size": sum(1 for a in assignments if a == j)}
                   for j in range(k)],
        "iterations_done": _ + 1,
    }


def list_clusters() -> list:
    conn = hw.get_conn()
    if conn is None:
        return []
    cur = conn.cursor()
    try:
        cur.execute("""
          SELECT id, label, project_slug, size, built_at
            FROM dashboard.clusters
           ORDER BY size DESC
        """)
        return [{"id": r[0], "label": r[1], "project_slug": r[2],
                 "size": r[3], "built_at": float(r[4])}
                for r in cur.fetchall()]
    except Exception:
        return []


def cluster_members(cluster_id: int, limit: int = 50) -> list:
    conn = hw.get_conn()
    if conn is None:
        return []
    cur = conn.cursor()
    cur.execute("""
      SELECT m.uuid, m.session_id, m.project_slug, m.timestamp,
             SUBSTR(m.prompt_text, 1, 160), cm.distance
        FROM dashboard.cluster_members cm
        JOIN dashboard.messages m ON m.uuid = cm.uuid
       WHERE cm.cluster_id = $1
       ORDER BY cm.distance ASC
       LIMIT $2
    """, (int(cluster_id), int(limit)))
    return [
        {"uuid": r[0], "session_id": r[1], "project_slug": r[2],
         "timestamp": r[3], "preview": r[4], "distance": float(r[5])}
        for r in cur.fetchall()
    ]


# ---------- helpers ----------

def _parse_vector(raw) -> list:
    if raw is None:
        return [0.0] * 384
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return [0.0] * 384


def _l2(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _cosine(a: list, b: list) -> float:
    """Returns 1 - cosine_similarity (so smaller = more similar)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return 1.0 - dot / (na * nb)
