from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from food_calorie_estimation.config import DataConfig

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_FIGURE_NAMES = (
    "train-source-groups.png",
    "train-calorie-distributions.png",
    "validation-reliability.png",
    "test-predicted-vs-actual.png",
    "test-residuals.png",
    "test-interval-coverage.png",
    "test-confidence-diagnostics.png",
    "test-pit-histograms.png",
)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _validation_reliability(observations: pd.DataFrame, probabilities: pd.DataFrame) -> tuple[np.ndarray, ...]:
    validation = observations.loc[observations.split.eq("validation")]
    usable = probabilities.loc[
        probabilities.sample_id.isin(validation.sample_id) & probabilities.class_name.ne("other_or_unknown")
    ]
    wide = usable.pivot(index="sample_id", columns="class_name", values="calibrated_probability")
    wide = wide.reindex(validation.sample_id)
    if wide.empty or wide.isna().any(axis=None):
        raise ValueError("validation probabilities are missing samples or classes")
    top_indices = wide.to_numpy().argmax(axis=1)
    confidence = wide.to_numpy().max(axis=1)
    predicted = wide.columns.to_numpy()[top_indices]
    actual = validation.food_class.to_numpy()
    correct = predicted == actual
    edges = np.linspace(0, 1, 11)
    bins = np.minimum(np.digitize(confidence, edges[1:-1]), 9)
    populated = np.array([index for index in range(10) if np.any(bins == index)])
    mean_confidence = np.array([confidence[bins == index].mean() for index in populated])
    accuracy = np.array([correct[bins == index].mean() for index in populated])
    counts = np.array([(bins == index).sum() for index in populated])
    return mean_confidence, accuracy, counts


def generate_figures(
    data_config: DataConfig,
    probability_path: Path,
    prediction_path: Path,
    metrics_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate split-labelled scientific diagnostics from frozen artifacts."""

    observations = pd.read_parquet(data_config.processed_path)
    probabilities = pd.read_parquet(probability_path)
    predictions = pd.read_parquet(prediction_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_observations = {"sample_id", "group_id", "split", "is_eligible", "food_class", "calories_kcal"}
    required_probabilities = {"sample_id", "class_name", "calibrated_probability"}
    required_predictions = {
        "model_id",
        "actual_calories_kcal",
        "estimated_calories_kcal",
        "predicted_class_probability",
        "prediction_entropy",
        "interval_lower_kcal",
        "interval_upper_kcal",
    }
    for frame, required, name in (
        (observations, required_observations, "processed observations"),
        (probabilities, required_probabilities, "class probabilities"),
        (predictions, required_predictions, "model predictions"),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = observations.loc[observations.is_eligible]
    train = eligible.loc[eligible.split.eq("train")]
    train_groups = train.drop_duplicates(["food_class", "group_id"])
    class_order = sorted(train_groups.food_class.unique())

    counts = train_groups.groupby("food_class").group_id.nunique().reindex(class_order)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(counts.index, counts.to_numpy(), color="#4472C4")
    axis.set(title="Train split: independent source groups by class", xlabel="Food class", ylabel="Source groups")
    axis.tick_params(axis="x", rotation=35)
    _save(figure, output_dir / _FIGURE_NAMES[0])

    values = [train_groups.loc[train_groups.food_class.eq(label), "calories_kcal"].to_numpy() for label in class_order]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(values, tick_labels=class_order, showmeans=True)
    axis.set(
        title="Train split: calorie targets per independent source group",
        xlabel="Food class",
        ylabel="Calories (kcal)",
    )
    axis.tick_params(axis="x", rotation=35)
    _save(figure, output_dir / _FIGURE_NAMES[1])

    confidence, accuracy, bin_counts = _validation_reliability(eligible, probabilities)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", label="Perfect calibration")
    axis.plot(confidence, accuracy, color="#C44E52", marker="o", label="Calibrated baseline")
    for x_value, y_value, count in zip(confidence, accuracy, bin_counts, strict=True):
        axis.annotate(f"n={count}", (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(
        title=f"Validation reliability (n={int(bin_counts.sum())})",
        xlabel="Mean predicted confidence",
        ylabel="Observed top-class accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    _save(figure, output_dir / _FIGURE_NAMES[2])

    mixture = predictions.loc[predictions.model_id.eq("M3")].copy()
    if mixture.empty:
        raise ValueError("model predictions require M3 rows")
    actual = mixture.actual_calories_kcal.to_numpy()
    estimated = mixture.estimated_calories_kcal.to_numpy()
    limits = (float(min(actual.min(), estimated.min())), float(max(actual.max(), estimated.max())))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, estimated, alpha=0.65, s=18, color="#4472C4")
    axis.plot(limits, limits, color="#666666", linestyle="--")
    axis.set(
        title=f"M3 test predictions (n={len(mixture)})",
        xlabel="Actual calories (kcal)",
        ylabel="Expected calories (kcal)",
    )
    _save(figure, output_dir / _FIGURE_NAMES[3])

    residual = estimated - actual
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(estimated, residual, alpha=0.65, s=18, color="#4472C4")
    axis.axhline(0, color="#666666", linestyle="--")
    axis.set(
        title=f"M3 test residuals (n={len(mixture)})",
        xlabel="Expected calories (kcal)",
        ylabel="Residual: predicted - actual (kcal)",
    )
    _save(figure, output_dir / _FIGURE_NAMES[4])

    model_metrics = metrics["models"]
    interval_models = [name for name in ("M2", "M3") if "interval_coverage" in model_metrics[name]]
    coverage = [100 * float(model_metrics[name]["interval_coverage"]) for name in interval_models]
    figure, axis = plt.subplots(figsize=(6, 5))
    bars = axis.bar(interval_models, coverage, color=["#DD8452", "#4472C4"][: len(interval_models)])
    axis.axhline(100 * float(metrics["interval_level"]), color="#666666", linestyle="--", label="Nominal")
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in coverage])
    axis.set(title="Test prediction-interval coverage", xlabel="Model", ylabel="Coverage (%)", ylim=(0, 105))
    axis.legend()
    _save(figure, output_dir / _FIGURE_NAMES[5])

    width = mixture.interval_upper_kcal.to_numpy() - mixture.interval_lower_kcal.to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(mixture.predicted_class_probability, np.abs(residual), alpha=0.65, s=18, color="#4472C4")
    axes[0].set(xlabel="Top-class probability", ylabel="Absolute error (kcal)", title="Confidence vs error")
    axes[1].scatter(mixture.prediction_entropy, width, alpha=0.65, s=18, color="#55A868")
    axes[1].set(xlabel="Prediction entropy", ylabel="90% interval width (kcal)", title="Entropy vs interval width")
    figure.suptitle(f"M3 test uncertainty diagnostics (n={len(mixture)})")
    _save(figure, output_dir / _FIGURE_NAMES[6])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, model_id in zip(axes, ("M2", "M3"), strict=True):
        histogram = metrics["diagnostics"][f"{model_id}_pit_histogram"]
        edges = np.asarray(histogram["bin_edges"], dtype=float)
        pit_counts = np.asarray(histogram["counts"], dtype=int)
        axis.bar(edges[:-1], pit_counts, width=np.diff(edges), align="edge", color="#4472C4", edgecolor="#FFFFFF")
        axis.axhline(len(mixture) / len(pit_counts), color="#666666", linestyle="--")
        axis.set(title=model_id, xlabel="PIT value", ylabel="Test rows")
    figure.suptitle("Test probability integral transform histograms")
    _save(figure, output_dir / _FIGURE_NAMES[7])

    return [output_dir / name for name in _FIGURE_NAMES]
