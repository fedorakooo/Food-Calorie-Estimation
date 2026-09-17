from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from food_calorie_estimation.calorie_model import (
    fit_empirical_distributions,
    simulate_mixture,
    summarize_simulations,
)
from food_calorie_estimation.config import DataConfig, ExperimentConfig, ModelConfig


def estimate_calories(
    data_config: DataConfig,
    model_config: ModelConfig,
    experiment: ExperimentConfig,
    probability_path: Path,
    output_path: Path,
    artifact_path: Path,
) -> None:
    observations = pd.read_parquet(data_config.processed_path)
    observations = observations.loc[observations.is_eligible].copy()
    train = observations.loc[observations.split == "train"]
    distributions = fit_empirical_distributions(
        train.calories_kcal.to_numpy(),
        train.food_class.to_numpy(),
        model_config.calorie_model.minimum_class_size,
    )
    probabilities = pd.read_parquet(probability_path)
    labels = tuple(x for x in probabilities.class_name.unique() if x != "other_or_unknown")
    wide = probabilities.pivot(index="sample_id", columns="class_name", values="calibrated_probability")
    wide = wide.reindex(columns=labels, fill_value=0.0)
    samples = simulate_mixture(wide.to_numpy(), labels, distributions, experiment.n_simulations, experiment.seed)
    summaries = pd.DataFrame(summarize_simulations(samples, experiment.interval_level), index=wide.index).reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries.to_parquet(output_path, index=False)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "distribution": "empirical",
        "classes": {label: len(values) for label, values in distributions.items()},
        "n_simulations": experiment.n_simulations,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
