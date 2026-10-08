from __future__ import annotations

import hashlib

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


def describe_empirical_distributions(distributions: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    """Return JSON-compatible fitted values and descriptive train statistics."""

    descriptions: dict[str, dict[str, object]] = {}
    for label, raw_values in sorted(distributions.items()):
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or not len(values):
            raise ValueError("empirical distributions must be non-empty one-dimensional arrays")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("empirical calories must be finite and non-negative")
        descriptions[label] = {
            "count": int(len(values)),
            "mean_kcal": float(values.mean()),
            "median_kcal": float(np.median(values)),
            "std_kcal": float(values.std()),
            "min_kcal": float(values.min()),
            "p05_kcal": float(np.quantile(values, 0.05)),
            "p25_kcal": float(np.quantile(values, 0.25)),
            "p75_kcal": float(np.quantile(values, 0.75)),
            "p95_kcal": float(np.quantile(values, 0.95)),
            "max_kcal": float(values.max()),
            "values_kcal": values.tolist(),
        }
    return descriptions


def normalize_supported_probabilities(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
) -> np.ndarray:
    """Validate and renormalize probability mass over fitted food classes."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if values.shape[1] != len(labels):
        raise ValueError("probability columns do not match labels")
    if len(set(labels)) != len(labels):
        raise ValueError("probability labels must be unique")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")

    available = np.array([label in distributions for label in labels])
    usable = values * available
    totals = usable.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every image needs probability mass on a supported class")
    return usable / totals


def prediction_warning_codes(
    normalized_probabilities: np.ndarray,
    supported_probability_mass: np.ndarray,
    low_confidence_threshold: float = 0.5,
) -> list[list[str]]:
    """Return stable warning codes for user-facing calorie predictions."""

    probabilities = np.asarray(normalized_probabilities, dtype=float)
    supported_mass = np.asarray(supported_probability_mass, dtype=float)
    if probabilities.ndim != 2 or supported_mass.ndim != 1:
        raise ValueError("warning inputs must be a probability matrix and a mass vector")
    if len(probabilities) != len(supported_mass):
        raise ValueError("warning inputs must have matching rows")
    if probabilities.shape[1] == 0:
        raise ValueError("warning probabilities require at least one class")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(supported_mass)):
        raise ValueError("warning inputs must be finite")
    if np.any(probabilities < 0) or np.any(supported_mass < 0) or np.any(supported_mass > 1 + 1e-9):
        raise ValueError("warning inputs must be valid probabilities")
    if not 0 < low_confidence_threshold <= 1:
        raise ValueError("low_confidence_threshold must be in (0, 1]")

    warnings: list[list[str]] = []
    for row, mass in zip(probabilities, supported_mass, strict=True):
        codes = ["research_only", "portion_and_recipe_unobserved"]
        if float(row.max()) < low_confidence_threshold:
            codes.append("low_classifier_confidence")
        if mass < 1 - 1e-9:
            codes.append("unsupported_class_probability_discarded")
        warnings.append(codes)
    return warnings


def mixture_moments(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate exact empirical-mixture means and variances for each image."""

    normalized = normalize_supported_probabilities(probabilities, labels, distributions)
    means = np.array([np.mean(distributions[label]) if label in distributions else 0.0 for label in labels])
    second_moments = np.array(
        [np.mean(np.square(distributions[label])) if label in distributions else 0.0 for label in labels]
    )
    mixture_means = normalized @ means
    variances = normalized @ second_moments - np.square(mixture_means)
    return mixture_means, np.maximum(variances, 0.0)


def hard_class_indices(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return top supported class indices and their renormalized probabilities."""

    normalized = normalize_supported_probabilities(probabilities, labels, distributions)
    indices = normalized.argmax(axis=1)
    return indices, normalized[np.arange(len(normalized)), indices]


def predict_hard_class_mean(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict the exact empirical mean of the top supported class."""

    indices, _ = hard_class_indices(probabilities, labels, distributions)
    return np.array([np.mean(distributions[labels[index]]) for index in indices])


def _row_generator(seed: int, row_index: int, row_keys: tuple[str, ...] | None) -> np.random.Generator:
    if row_keys is None:
        sequence = np.random.SeedSequence([seed, row_index])
    else:
        digest = hashlib.sha256(row_keys[row_index].encode("utf-8")).digest()
        stable_key = int.from_bytes(digest[:8], byteorder="big", signed=False)
        sequence = np.random.SeedSequence([seed, stable_key])
    return np.random.default_rng(sequence)


def simulate_hard_class_distribution(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
    n_simulations: int,
    seed: int,
    row_keys: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Bootstrap calories from the top supported class for each image."""

    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    indices, _ = hard_class_indices(probabilities, labels, distributions)
    if row_keys is not None and len(row_keys) != len(indices):
        raise ValueError("row_keys must match the number of probability rows")
    return np.vstack(
        [
            _row_generator(seed, row, row_keys).choice(distributions[labels[index]], size=n_simulations, replace=True)
            for row, index in enumerate(indices)
        ]
    )


def simulate_mixture(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    distributions: dict[str, np.ndarray],
    n_simulations: int,
    seed: int,
    row_keys: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Sample a class-probability mixture with bootstrap empirical outcomes."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    normalized = normalize_supported_probabilities(probabilities, labels, distributions)
    if row_keys is not None and len(row_keys) != len(normalized):
        raise ValueError("row_keys must match the number of probability rows")
    output = np.zeros((len(probabilities), n_simulations), dtype=float)
    for row, weights in enumerate(normalized):
        rng = _row_generator(seed, row, row_keys)
        chosen = rng.choice(len(labels), size=n_simulations, p=weights)
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
            "variance_calories_kcal2": float(np.var(row)),
            "std_calories_kcal": float(np.std(row)),
            "interval_lower_kcal": float(np.quantile(row, alpha)),
            "interval_upper_kcal": float(np.quantile(row, 1 - alpha)),
            "p05_kcal": float(np.quantile(row, 0.05)),
            "p50_kcal": float(np.quantile(row, 0.50)),
            "p95_kcal": float(np.quantile(row, 0.95)),
        }
        for row in samples
    ]


def evaluate_point_estimates(actual: np.ndarray, estimated: np.ndarray) -> dict[str, float]:
    """Calculate point-prediction diagnostics for held-out observations."""

    actual_values = np.asarray(actual, dtype=float)
    estimated_values = np.asarray(estimated, dtype=float)
    if actual_values.ndim != 1 or estimated_values.ndim != 1:
        raise ValueError("point-evaluation inputs must be one-dimensional arrays")
    if not len(actual_values) or len(actual_values) != len(estimated_values):
        raise ValueError("point-evaluation inputs must be non-empty and have equal lengths")
    if not np.all(np.isfinite(actual_values)) or not np.all(np.isfinite(estimated_values)):
        raise ValueError("point-evaluation inputs must be finite")
    if np.any(actual_values < 0) or np.any(estimated_values < 0):
        raise ValueError("actual and estimated calories must be non-negative")
    error = estimated_values - actual_values
    return {
        "mae_kcal": float(np.abs(error).mean()),
        "median_absolute_error_kcal": float(np.median(np.abs(error))),
        "rmse_kcal": float(np.sqrt(np.mean(error**2))),
        "mean_error_kcal": float(error.mean()),
    }


def empirical_crps(actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Calculate CRPS per row from finite Monte Carlo samples in O(n log n)."""

    actual_values = np.asarray(actual, dtype=float)
    sample_values = np.asarray(samples, dtype=float)
    if actual_values.ndim != 1 or sample_values.ndim != 2:
        raise ValueError("CRPS requires one-dimensional actuals and two-dimensional samples")
    if not len(actual_values) or len(actual_values) != len(sample_values) or sample_values.shape[1] == 0:
        raise ValueError("CRPS inputs must be non-empty and have matching rows")
    if not np.all(np.isfinite(actual_values)) or not np.all(np.isfinite(sample_values)):
        raise ValueError("CRPS inputs must be finite")
    ordered = np.sort(sample_values, axis=1)
    count = ordered.shape[1]
    coefficients = 2 * np.arange(1, count + 1) - count - 1
    pairwise_term = (ordered * coefficients).sum(axis=1) / count**2
    return np.abs(sample_values - actual_values[:, None]).mean(axis=1) - pairwise_term


def evaluate_interval_predictions(
    actual: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    interval_level: float,
) -> dict[str, float]:
    """Calculate coverage, sharpness, and the proper central interval score."""

    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be between 0 and 1")
    actual_values, lower_values, upper_values = (
        np.asarray(values, dtype=float) for values in (actual, interval_lower, interval_upper)
    )
    if any(values.ndim != 1 for values in (actual_values, lower_values, upper_values)):
        raise ValueError("interval-evaluation inputs must be one-dimensional arrays")
    if not len(actual_values) or any(len(values) != len(actual_values) for values in (lower_values, upper_values)):
        raise ValueError("interval-evaluation inputs must be non-empty and have equal lengths")
    if not all(np.all(np.isfinite(values)) for values in (actual_values, lower_values, upper_values)):
        raise ValueError("interval-evaluation inputs must be finite")
    if np.any(lower_values > upper_values):
        raise ValueError("interval lower bounds cannot exceed upper bounds")

    width = upper_values - lower_values
    alpha = 1 - interval_level
    score = width.copy()
    score += (2 / alpha) * (lower_values - actual_values) * (actual_values < lower_values)
    score += (2 / alpha) * (actual_values - upper_values) * (actual_values > upper_values)
    return {
        "interval_coverage": float(((actual_values >= lower_values) & (actual_values <= upper_values)).mean()),
        "mean_interval_width_kcal": float(width.mean()),
        "median_interval_width_kcal": float(np.median(width)),
        "mean_interval_score_kcal": float(score.mean()),
    }


def evaluate_estimates(
    actual: np.ndarray,
    estimated: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    interval_level: float = 0.9,
) -> dict[str, float]:
    """Calculate point and interval diagnostics for held-out calorie estimates."""

    arrays = tuple(np.asarray(values, dtype=float) for values in (actual, estimated, interval_lower, interval_upper))
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("evaluation inputs must be one-dimensional arrays")
    if not len(arrays[0]) or any(len(values) != len(arrays[0]) for values in arrays[1:]):
        raise ValueError("evaluation inputs must be non-empty and have equal lengths")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("evaluation inputs must be finite")
    actual_values, estimated_values, lower_values, upper_values = arrays
    if np.any(actual_values < 0) or np.any(estimated_values < 0):
        raise ValueError("actual and estimated calories must be non-negative")
    if np.any(lower_values > upper_values):
        raise ValueError("interval lower bounds cannot exceed upper bounds")

    return {
        **evaluate_point_estimates(actual_values, estimated_values),
        **evaluate_interval_predictions(actual_values, lower_values, upper_values, interval_level),
    }
