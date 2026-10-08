from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from food_calorie_estimation.calorie_model import (
    empirical_crps,
    evaluate_interval_predictions,
    evaluate_point_estimates,
    fit_empirical_distributions,
    hard_class_indices,
    mixture_moments,
    normalize_supported_probabilities,
    predict_hard_class_mean,
    simulate_hard_class_distribution,
    simulate_mixture,
    summarize_simulations,
)
from food_calorie_estimation.config import DataConfig, ExperimentConfig, ModelConfig

_OBSERVATION_COLUMNS = {"sample_id", "group_id", "split", "is_eligible", "food_class", "calories_kcal"}
_PROBABILITY_COLUMNS = {"sample_id", "class_name", "calibrated_probability"}
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


def _require_columns(frame: pd.DataFrame, required: set[str], artifact_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact_name} is missing required columns: {missing}")


def _sample_summary(
    samples: np.ndarray,
    expected: np.ndarray,
    variance: np.ndarray,
    interval_level: float,
) -> pd.DataFrame:
    summary = pd.DataFrame(summarize_simulations(samples, interval_level))
    summary["simulation_mean_calories_kcal"] = summary["estimated_calories_kcal"]
    summary["estimated_calories_kcal"] = expected
    summary["variance_calories_kcal2"] = variance
    summary["std_calories_kcal"] = np.sqrt(variance)
    return summary


def _probabilistic_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    summary: pd.DataFrame,
    samples: np.ndarray,
    interval_level: float,
) -> dict[str, float]:
    return {
        **evaluate_point_estimates(actual, expected),
        **evaluate_interval_predictions(
            actual,
            summary.interval_lower_kcal.to_numpy(),
            summary.interval_upper_kcal.to_numpy(),
            interval_level,
        ),
        "mean_crps_kcal": float(empirical_crps(actual, samples).mean()),
    }


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first_ranks = pd.Series(first).rank(method="average").to_numpy()
    second_ranks = pd.Series(second).rank(method="average").to_numpy()
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _cluster_bootstrap(
    group_ids: np.ndarray,
    seed: int,
    statistic: object,
) -> tuple[float, float] | None:
    """Return a percentile CI while resampling independent source groups."""

    groups = np.unique(group_ids)
    if len(groups) < 2:
        return None
    positions = [np.flatnonzero(group_ids == group) for group in groups]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([positions[index] for index in selected])
        value = statistic(indices)  # type: ignore[operator]
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    alpha = (1 - _BOOTSTRAP_CONFIDENCE_LEVEL) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def _interval_payload(interval: tuple[float, float] | None) -> dict[str, float] | None:
    if interval is None:
        return None
    return {"lower": interval[0], "upper": interval[1]}


def _prediction_bootstrap_intervals(
    group_ids: np.ndarray,
    actual: np.ndarray,
    estimated: np.ndarray,
    summary: pd.DataFrame | None,
    crps: np.ndarray | None,
    seed: int,
) -> dict[str, dict[str, float] | None]:
    error = estimated - actual
    statistics: dict[str, object] = {
        "mae_kcal": lambda indices: np.abs(error[indices]).mean(),
        "rmse_kcal": lambda indices: np.sqrt(np.mean(error[indices] ** 2)),
        "mean_error_kcal": lambda indices: error[indices].mean(),
    }
    if summary is not None:
        lower = summary.interval_lower_kcal.to_numpy()
        upper = summary.interval_upper_kcal.to_numpy()
        covered = (actual >= lower) & (actual <= upper)
        width = upper - lower
        statistics.update(
            {
                "interval_coverage": lambda indices: covered[indices].mean(),
                "mean_interval_width_kcal": lambda indices: width[indices].mean(),
            }
        )
    if crps is not None:
        statistics["mean_crps_kcal"] = lambda indices: crps[indices].mean()
    return {
        name: _interval_payload(_cluster_bootstrap(group_ids, seed + offset, statistic))
        for offset, (name, statistic) in enumerate(statistics.items())
    }


def _pit_histogram(actual: np.ndarray, samples: np.ndarray) -> dict[str, object]:
    pit = (samples <= actual[:, None]).mean(axis=1)
    counts, edges = np.histogram(pit, bins=np.linspace(0, 1, 11))
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def _confidence_diagnostics(
    actual: np.ndarray,
    estimated: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    summary: pd.DataFrame,
    group_ids: np.ndarray,
    seed: int,
) -> dict[str, object]:
    absolute_error = np.abs(estimated - actual)
    width = summary.interval_upper_kcal.to_numpy() - summary.interval_lower_kcal.to_numpy()
    covered = (actual >= summary.interval_lower_kcal.to_numpy()) & (actual <= summary.interval_upper_kcal.to_numpy())
    frame = pd.DataFrame(
        {
            "confidence": confidence,
            "absolute_error_kcal": absolute_error,
            "interval_width_kcal": width,
            "covered": covered,
        }
    )
    frame["confidence_decile"] = pd.qcut(frame.confidence, q=10, labels=False, duplicates="drop")
    deciles = [
        {
            "confidence_decile": int(decile),
            "rows": int(len(group)),
            "mean_confidence": float(group.confidence.mean()),
            "mean_absolute_error_kcal": float(group.absolute_error_kcal.mean()),
            "mean_interval_width_kcal": float(group.interval_width_kcal.mean()),
            "interval_coverage": float(group.covered.mean()),
        }
        for decile, (_, group) in enumerate(frame.groupby("confidence_decile", observed=True))
    ]
    correlations = {
        "top_probability_vs_absolute_error_spearman": _spearman(confidence, absolute_error),
        "entropy_vs_absolute_error_spearman": _spearman(entropy, absolute_error),
        "entropy_vs_interval_width_spearman": _spearman(entropy, width),
    }
    correlation_pairs = {
        "top_probability_vs_absolute_error_spearman": (confidence, absolute_error),
        "entropy_vs_absolute_error_spearman": (entropy, absolute_error),
        "entropy_vs_interval_width_spearman": (entropy, width),
    }
    bootstrap_intervals = {
        name: _interval_payload(
            _cluster_bootstrap(
                group_ids,
                seed + offset,
                lambda indices, pair=pair: _spearman(pair[0][indices], pair[1][indices]),
            )
        )
        for offset, (name, pair) in enumerate(correlation_pairs.items())
    }
    return {
        **correlations,
        "bootstrap_95_ci": bootstrap_intervals,
        "confidence_deciles": deciles,
    }


def _per_class_metrics(
    actual_classes: np.ndarray,
    actual: np.ndarray,
    estimated: np.ndarray,
    summary: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    lower = summary.interval_lower_kcal.to_numpy()
    upper = summary.interval_upper_kcal.to_numpy()
    for label in sorted(set(actual_classes.tolist())):
        selected = actual_classes == label
        errors = estimated[selected] - actual[selected]
        result[label] = {
            "rows": int(selected.sum()),
            "mae_kcal": float(np.abs(errors).mean()),
            "rmse_kcal": float(np.sqrt(np.mean(errors**2))),
            "interval_coverage": float(
                ((actual[selected] >= lower[selected]) & (actual[selected] <= upper[selected])).mean()
            ),
        }
    return result


def compare_models(
    data_config: DataConfig,
    model_config: ModelConfig,
    experiment: ExperimentConfig,
    probability_path: Path,
    prediction_path: Path,
    metrics_path: Path,
) -> None:
    """Compare M1/M2/M3 once on the locked, eligible test split."""

    observations = pd.read_parquet(data_config.processed_path)
    probabilities = pd.read_parquet(probability_path)
    _require_columns(observations, _OBSERVATION_COLUMNS, "processed observations")
    _require_columns(probabilities, _PROBABILITY_COLUMNS, "class probabilities")
    if observations.sample_id.duplicated().any():
        raise ValueError("processed observations must contain unique sample_id values")
    if probabilities.duplicated(["sample_id", "class_name"]).any():
        raise ValueError("class probabilities must contain one row per sample_id/class_name")

    eligible = observations.loc[observations.is_eligible].copy()
    train = eligible.loc[eligible.split.eq("train")]
    test = eligible.loc[eligible.split.eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("processed observations require non-empty train and test splits")
    distributions = fit_empirical_distributions(
        train.calories_kcal.to_numpy(),
        train.food_class.to_numpy(),
        model_config.calorie_model.minimum_class_size,
    )

    labels = tuple(name for name in probabilities.class_name.drop_duplicates() if name != "other_or_unknown")
    if not labels:
        raise ValueError("class probabilities require at least one food class")
    wide = probabilities.pivot(index="sample_id", columns="class_name", values="calibrated_probability")
    wide = wide.reindex(index=test.sample_id, columns=labels, fill_value=0.0)
    if wide.isna().any(axis=None):
        missing = wide.index[wide.isna().any(axis=1)].tolist()
        raise ValueError(f"class probabilities are missing held-out test samples or classes: {missing}")
    probability_values = wide.to_numpy()
    row_keys = tuple(wide.index.astype(str))
    actual = test.calories_kcal.to_numpy(dtype=float)
    group_ids = test.group_id.to_numpy()
    normalized_probabilities = normalize_supported_probabilities(probability_values, labels, distributions)
    entropy = -(normalized_probabilities * np.log(np.clip(normalized_probabilities, 1e-12, 1.0))).sum(axis=1)

    hard_indices, hard_probabilities = hard_class_indices(probability_values, labels, distributions)
    hard_labels = np.array(labels)[hard_indices]
    hard_mean = predict_hard_class_mean(probability_values, labels, distributions)
    hard_variance = np.array([np.var(distributions[label]) for label in hard_labels])
    hard_samples = simulate_hard_class_distribution(
        probability_values,
        labels,
        distributions,
        experiment.n_simulations,
        experiment.seed,
        row_keys,
    )
    hard_summary = _sample_summary(hard_samples, hard_mean, hard_variance, experiment.interval_level)
    hard_crps = empirical_crps(actual, hard_samples)

    mixture_mean, mixture_variance = mixture_moments(probability_values, labels, distributions)
    mixture_samples = simulate_mixture(
        probability_values,
        labels,
        distributions,
        experiment.n_simulations,
        experiment.seed,
        row_keys,
    )
    mixture_summary = _sample_summary(
        mixture_samples,
        mixture_mean,
        mixture_variance,
        experiment.interval_level,
    )
    mixture_crps = empirical_crps(actual, mixture_samples)

    common = pd.DataFrame(
        {
            "sample_id": test.sample_id.to_numpy(),
            "group_id": group_ids,
            "actual_calories_kcal": actual,
            "actual_class": test.food_class.to_numpy(),
            "predicted_class": hard_labels,
            "predicted_class_probability": hard_probabilities,
            "prediction_entropy": entropy,
        }
    )
    mean_predictions = common.assign(
        model_id="M1",
        model_name="hard_class_mean",
        estimated_calories_kcal=hard_mean,
        variance_calories_kcal2=np.nan,
        std_calories_kcal=np.nan,
        interval_lower_kcal=np.nan,
        interval_upper_kcal=np.nan,
        p05_kcal=np.nan,
        p50_kcal=hard_mean,
        p95_kcal=np.nan,
        simulation_mean_calories_kcal=np.nan,
        crps_kcal=np.nan,
    )
    hard_predictions = pd.concat(
        [
            common.assign(model_id="M2", model_name="hard_class_distribution").reset_index(drop=True),
            hard_summary.reset_index(drop=True),
        ],
        axis=1,
    )
    hard_predictions["crps_kcal"] = hard_crps
    mixture_predictions = pd.concat(
        [
            common.assign(model_id="M3", model_name="probability_mixture").reset_index(drop=True),
            mixture_summary.reset_index(drop=True),
        ],
        axis=1,
    )
    mixture_predictions["crps_kcal"] = mixture_crps
    predictions = pd.concat([mean_predictions, hard_predictions, mixture_predictions], ignore_index=True)

    metrics: dict[str, object] = {
        "evaluation_split": "test",
        "test_rows": int(len(test)),
        "interval_level": experiment.interval_level,
        "n_simulations": experiment.n_simulations,
        "seed": experiment.seed,
        "calorie_training_split": "train",
        "calorie_sampling_unit": "eligible image-object row",
        "training_source_groups_by_class": {
            label: int(train.loc[train.food_class.eq(label), "group_id"].nunique()) for label in sorted(distributions)
        },
        "supported_classes": sorted(distributions),
        "models": {
            "M1": {
                "name": "hard_class_mean",
                "uncertainty": "none",
                **evaluate_point_estimates(actual, hard_mean),
                "bootstrap_95_ci": _prediction_bootstrap_intervals(
                    group_ids, actual, hard_mean, None, None, experiment.seed
                ),
            },
            "M2": {
                "name": "hard_class_distribution",
                "uncertainty": "top-class empirical bootstrap",
                **_probabilistic_metrics(actual, hard_mean, hard_summary, hard_samples, experiment.interval_level),
                "bootstrap_95_ci": _prediction_bootstrap_intervals(
                    group_ids, actual, hard_mean, hard_summary, hard_crps, experiment.seed + 100
                ),
            },
            "M3": {
                "name": "probability_mixture",
                "uncertainty": "class-probability empirical mixture",
                **_probabilistic_metrics(
                    actual,
                    mixture_mean,
                    mixture_summary,
                    mixture_samples,
                    experiment.interval_level,
                ),
                "bootstrap_95_ci": _prediction_bootstrap_intervals(
                    group_ids, actual, mixture_mean, mixture_summary, mixture_crps, experiment.seed + 200
                ),
            },
        },
        "bootstrap": {
            "method": "percentile cluster bootstrap over test source groups",
            "confidence_level": _BOOTSTRAP_CONFIDENCE_LEVEL,
            "resamples": _BOOTSTRAP_RESAMPLES,
            "seed": experiment.seed,
            "test_source_groups": int(len(np.unique(group_ids))),
        },
        "diagnostics": {
            "M2_pit_histogram": _pit_histogram(actual, hard_samples),
            "M3_pit_histogram": _pit_histogram(actual, mixture_samples),
            "M3_confidence_and_error": _confidence_diagnostics(
                actual,
                mixture_mean,
                hard_probabilities,
                entropy,
                mixture_summary,
                group_ids,
                experiment.seed + 300,
            ),
            "M3_per_actual_class": _per_class_metrics(
                test.food_class.to_numpy(),
                actual,
                mixture_mean,
                mixture_summary,
            ),
        },
    }
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
