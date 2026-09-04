"""Interval estimates and paired tests for accuracy figures and tables."""

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")

    low, high = proportion_confint(count=successes, nobs=n, alpha=alpha, method="wilson")
    return float(low), float(high)


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """
    Exact McNemar test for two conditions scored on the *same* items.

    The right test whenever conditions share their questions, which is every comparison in
    this project: retrieval configurations rank the same 778 questions, and the end-to-end
    conditions answer them. Marginal Wilson intervals are the wrong tool there -- they
    routinely overlap on differences a paired test resolves at p < 1e-4, because they throw
    away the pairing that makes the comparison sensitive.

    Only discordant pairs carry information: items both conditions get right, or both get
    wrong, say nothing about which is better.

    Args:
        a: per-item boolean outcomes for one condition.
        b: the same items, same order, for the other.

    Returns:
        `delta` (a - b, in proportion), `n10` (a right, b wrong), `n01` (a wrong, b right),
        `n`, and the two-sided exact `p`.

    Raises:
        ValueError: if the two conditions do not cover the same number of items -- a length
            mismatch means they are not paired and the test does not apply.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired outcomes must align, got {a.shape} and {b.shape}")

    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "delta": float(a.mean() - b.mean()),
        "n10": n10,
        "n01": n01,
        "n": int(a.size),
        "p": float(mcnemar([[0, n10], [n01, 0]], exact=True).pvalue),
    }
