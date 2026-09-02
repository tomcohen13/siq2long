"""Interval estimates for accuracy figures and tables."""

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
