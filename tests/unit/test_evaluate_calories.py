import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from food_calorie_estimation.calorie_model import evaluate_estimates
from food_calorie_estimation.config import DataConfig, ExperimentConfig
from food_calorie_estimation.pipelines.evaluate_calories import evaluate_calories


def test_evaluate_estimates_reports_point_and_interval_metrics() -> None:
    metrics = evaluate_estimates(
        np.array([100.0, 200.0]),
        np.array([110.0, 190.0]),
        np.array([90.0, 170.0]),
        np.array([120.0, 210.0]),
    )

    assert metrics == {
        "mae_kcal": 10.0,
        "rmse_kcal": 10.0,
        "mean_error_kcal": 0.0,
        "interval_coverage": 1.0,
        "mean_interval_width_kcal": 35.0,
    }


def test_evaluate_estimates_rejects_invalid_intervals() -> None:
    with pytest.raises(ValueError, match="lower bounds"):
        evaluate_estimates(np.array([100.0]), np.array([100.0]), np.array([110.0]), np.array([90.0]))


def test_evaluation_uses_only_held_out_test_rows(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.parquet"
    estimates_path = tmp_path / "estimates.parquet"
    output_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "sample_id": ["train", "test-a", "test-b"],
            "split": ["train", "test", "test"],
            "is_eligible": [True, True, True],
            "calories_kcal": [999.0, 100.0, 200.0],
        }
    ).to_parquet(observations_path, index=False)
    pd.DataFrame(
        {
            "sample_id": ["train", "test-a", "test-b"],
            "estimated_calories_kcal": [1.0, 110.0, 190.0],
            "interval_lower_kcal": [0.0, 90.0, 170.0],
            "interval_upper_kcal": [2.0, 120.0, 210.0],
        }
    ).to_parquet(estimates_path, index=False)
    data_config = DataConfig(
        dataset_name="test",
        raw_root=tmp_path,
        images_dir=Path("images"),
        annotations_dir=Path("annotations"),
        energy_density_path=tmp_path / "density.yaml",
        processed_path=observations_path,
        seed=1,
        minimum_class_size=2,
    )
    experiment = ExperimentConfig.model_validate(
        {
            "run_name": "test-run",
            "seed": 1,
            "interval_level": 0.9,
            "n_simulations": 10,
            "splits": {"train": 0.7, "validation": 0.15, "test": 0.15},
        }
    )

    evaluate_calories(data_config, experiment, estimates_path, output_path)

    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["test_rows"] == 2
    assert metrics["mae_kcal"] == 10.0
    assert metrics["rmse_kcal"] == 10.0
    assert metrics["interval_coverage"] == 1.0
