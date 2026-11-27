"""Pure metric functions (no I/O) so they are unit-testable."""
from __future__ import annotations

import math
import random


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den else None


def pairwise_outcomes(items):
    """items: iterable of (group, gold_tier, score). Returns list of 1.0 / 0.5 / 0.0 per ordered gold pair."""
    by_group: dict = {}
    for g, tier, sc in items:
        by_group.setdefault(g, []).append((tier, sc))
    out = []
    for rows in by_group.values():
        for ti, si in rows:
            for tj, sj in rows:
                if ti > tj:
                    out.append(1.0 if si > sj else 0.5 if si == sj else 0.0)
    return out


def bootstrap_ci(values, n=1000, seed=0, alpha=0.05):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return [round(means[int(n * alpha / 2)], 4), round(means[int(n * (1 - alpha / 2)) - 1], 4)]


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def r4(x):
    return None if x is None else round(x, 4)
