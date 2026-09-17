from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from food_calorie_estimation.calorie_model import evaluate_estimates
from food_calorie_estimation.config import DataConfig, ExperimentConfig

_OBSERVATION_COLUMNS = {"sample_id", "split", "is_eligible", "calories_kcal"}
_ESTIMATE_COLUMNS = {
    "sample_id",
    "estimated_calories_kcal",
    "interval_lower_kcal",
    "interval_upper_kcal",
}


def _require_columns(frame: pd.DataFrame, required: set[str], artifact_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact_name} is missing required columns: {missing}")


def evaluate_calories(
    data_config: DataConfig,
    experiment: ExperimentConfig,
    estimate_path: Path,
    output_path: Path,
) -> None:
    """Evaluate calorie estimates exclusively against held-out eligible test rows."""

    observations = pd.read_parquet(data_config.processed_path)
    estimates = pd.read_parquet(estimate_path)
    _require_columns(observations, _OBSERVATION_COLUMNS, "processed observations")
    _require_columns(estimates, _ESTIMATE_COLUMNS, "calorie estimates")
    if estimates.sample_id.duplicated().any():
        raise ValueError("calorie estimates must contain one row per sample_id")

    test_rows = observations.loc[
        observations.is_eligible & observations.split.eq("test"),
        ["sample_id", "calories_kcal"],
    ]
    if test_rows.empty:
        raise ValueError("processed observations require at least one eligible test row")
    evaluated = test_rows.merge(estimates, on="sample_id", how="left", validate="one_to_one")
    if evaluated.estimated_calories_kcal.isna().any():
        missing = evaluated.loc[evaluated.estimated_calories_kcal.isna(), "sample_id"].tolist()
        raise ValueError(f"calorie estimates are missing held-out test samples: {missing}")

    metrics: dict[str, float | int | str] = {
        **evaluate_estimates(
            evaluated.calories_kcal.to_numpy(),
            evaluated.estimated_calories_kcal.to_numpy(),
            evaluated.interval_lower_kcal.to_numpy(),
            evaluated.interval_upper_kcal.to_numpy(),
        ),
        "evaluation_split": "test",
        "test_rows": int(len(evaluated)),
        "interval_level": experiment.interval_level,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
