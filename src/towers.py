"""Two-tower toy with cold-start evaluation slices."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

NEWS = {
    "n1": "eu battery recycling",
    "n2": "turin rental market",
    "n3": "column ranking v4",
    "n4": "stellantis service logs",
    "n5": "ego4d nlq video",
}

# user, clicked, candidate, is_new_user, is_new_item
IMPR = [
    ("u1", ["n1", "n3"], "n3", False, False),
    ("u1", ["n1", "n3"], "n4", False, True),
    ("u2", [], "n2", True, False),
    ("u3", [], "n5", True, True),
    ("u2", [], "n1", True, False),
]


def embed(text: str, dim: int = 16) -> list[float]:
    v = [0.0] * dim
    for tok in text.split():
        h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def user_vec(history: list[str]) -> list[float]:
    if not history:
        return embed("UNK USER")
    acc = [0.0] * 16
    for nid in history:
        e = embed(NEWS[nid])
        acc = [a + b for a, b in zip(acc, e)]
    n = math.sqrt(sum(x * x for x in acc)) or 1.0
    return [x / n for x in acc]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def recall_by_slice() -> dict[str, float]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for user, hist, cand, new_u, new_i in IMPR:
        uv = user_vec(hist)
        ranked = sorted(NEWS, key=lambda nid: -dot(uv, embed(NEWS[nid])))
        hit = int(cand in ranked[:2])
        buckets["all"].append(hit)
        if new_u and new_i:
            buckets["both_new"].append(hit)
        elif new_u:
            buckets["new_user"].append(hit)
        elif new_i:
            buckets["new_item"].append(hit)
        else:
            buckets["warm"].append(hit)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


if __name__ == "__main__":
    for k, v in recall_by_slice().items():
        print(f"{k:10s} recall@2={v:.2f}")
