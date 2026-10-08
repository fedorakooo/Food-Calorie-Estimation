from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from food_calorie_estimation.calorie_model import (
    describe_empirical_distributions,
    fit_empirical_distributions,
    mixture_moments,
    normalize_supported_probabilities,
    prediction_warning_codes,
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
    probability_values = wide.to_numpy()
    supported = pd.Series([label in distributions for label in labels], index=labels).to_numpy()
    supported_mass = (probability_values * supported).sum(axis=1)
    normalized = normalize_supported_probabilities(probability_values, labels, distributions)
    row_keys = tuple(wide.index.astype(str))
    samples = simulate_mixture(
        probability_values, labels, distributions, experiment.n_simulations, experiment.seed, row_keys
    )
    expected, variance = mixture_moments(probability_values, labels, distributions)
    summaries = pd.DataFrame(summarize_simulations(samples, experiment.interval_level), index=wide.index).reset_index()
    summaries["simulation_mean_calories_kcal"] = summaries["estimated_calories_kcal"]
    summaries["estimated_calories_kcal"] = expected
    summaries["variance_calories_kcal2"] = variance
    summaries["std_calories_kcal"] = variance**0.5
    summaries["supported_probability_mass"] = supported_mass
    summaries["warning_codes"] = [";".join(codes) for codes in prediction_warning_codes(normalized, supported_mass)]
    summaries["n_simulations"] = experiment.n_simulations
    summaries["interval_level"] = experiment.interval_level
    summaries["seed"] = experiment.seed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries.to_parquet(output_path, index=False)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    class_descriptions = describe_empirical_distributions(distributions)
    for label, description in class_descriptions.items():
        class_rows = train.loc[train.food_class.eq(label)]
        description["source_groups"] = int(class_rows.group_id.nunique())
        description["unique_targets"] = int(class_rows.calories_kcal.nunique())
    artifact = {
        "model_version": "empirical-bootstrap-v1",
        "distribution": "empirical",
        "fitted_split": "train",
        "small_class_policy": model_config.calorie_model.small_class_policy,
        "minimum_class_size": model_config.calorie_model.minimum_class_size,
        "sampling_unit": "eligible image-object row",
        "classes": class_descriptions,
        "n_simulations": experiment.n_simulations,
        "seed": experiment.seed,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
