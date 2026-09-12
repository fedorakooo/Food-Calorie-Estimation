from __future__ import annotations

import numpy as np


def fit_empirical_distributions(
    calories: np.ndarray, labels: np.ndarray, minimum_class_size: int
) -> dict[str, np.ndarray]:
    """Fit class-conditional empirical calorie distributions."""
    result: dict[str, np.ndarray] = {}
    for label in sorted(set(labels.tolist())):
        values = np.asarray(calories[labels == label], dtype=float)
        if len(values) < minimum_class_size:
            continue
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("training calories must be finite and non-negative")
        result[label] = values
    if not result:
        raise ValueError("no class has enough training observations")
    return result


def simulate_mixture(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    """Sample a class-probability mixture with bootstrap empirical outcomes."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if probabilities.shape[1] != len(labels):
        raise ValueError("probability columns do not match labels")
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    rng = np.random.default_rng(seed)
    output = np.zeros((len(probabilities), n_simulations), dtype=float)
    for row, weights in enumerate(probabilities):
        available = np.array([label in distributions for label in labels])
        usable = weights * available
        total = usable.sum()
        if total <= 0:
            raise ValueError("every image needs probability mass on a supported class")
        usable /= total
        chosen = rng.choice(len(labels), size=n_simulations, p=usable)
        for index in np.unique(chosen):
            values = distributions[labels[index]]
            mask = chosen == index
            output[row, mask] = rng.choice(values, size=int(mask.sum()), replace=True)
    return output


def summarize_simulations(samples: np.ndarray, interval_level: float) -> list[dict[str, float]]:
    """Summarize simulations using a central interval and stable quantiles."""
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("samples must have at least one simulation per row")
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be between 0 and 1")
    alpha = (1 - interval_level) / 2
    return [
        {
            "estimated_calories_kcal": float(np.mean(row)),
            "interval_lower_kcal": float(np.quantile(row, alpha)),
            "interval_upper_kcal": float(np.quantile(row, 1 - alpha)),
            "p05_kcal": float(np.quantile(row, 0.05)),
            "p50_kcal": float(np.quantile(row, 0.50)),
            "p95_kcal": float(np.quantile(row, 0.95)),
        }
        for row in samples
    ]
