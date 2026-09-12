from __future__ import annotations

import json
from pathlib import Path

from food_calorie_estimation.config import DataConfig, ExperimentConfig
from food_calorie_estimation.data.ingest import build_observations
from food_calorie_estimation.data.split import assign_group_safe_splits


def prepare_data(
    config: DataConfig,
    experiment: ExperimentConfig,
    audit_path: Path,
    split_path: Path,
) -> None:
    """Create the canonical table and audit report for a configured data source."""

    observations, report = build_observations(config)
    eligible = observations.is_eligible
    if eligible.any():
        observations.loc[eligible, "split"] = assign_group_safe_splits(
            observations.loc[eligible], experiment.splits, experiment.seed
        )
        report.split_counts.update(observations.loc[eligible, "split"])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        observations.loc[eligible, ["sample_id", "group_id", "split"]].to_csv(
            split_path, index=False
        )
    config.processed_path.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(config.processed_path, index=False)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
