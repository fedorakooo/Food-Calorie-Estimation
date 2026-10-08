import json
from pathlib import Path

import pandas as pd

from food_calorie_estimation.config import DataConfig, ExperimentConfig, ModelConfig
from food_calorie_estimation.pipelines.compare_models import compare_models


def test_compare_models_writes_frozen_test_metrics_and_predictions(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.parquet"
    probabilities_path = tmp_path / "probabilities.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "sample_id": ["train-a1", "train-a2", "train-b1", "train-b2", "test-a", "test-b"],
            "group_id": ["train-a1", "train-a2", "train-b1", "train-b2", "test-a", "test-b"],
            "split": ["train", "train", "train", "train", "test", "test"],
            "is_eligible": [True] * 6,
            "food_class": ["apple", "apple", "banana", "banana", "apple", "banana"],
            "calories_kcal": [10.0, 20.0, 30.0, 40.0, 12.0, 38.0],
        }
    ).to_parquet(observations_path, index=False)
    pd.DataFrame(
        [
            {"sample_id": sample_id, "class_name": class_name, "calibrated_probability": probability}
            for sample_id, values in {
                "test-a": {"apple": 0.8, "banana": 0.2, "other_or_unknown": 0.0},
                "test-b": {"apple": 0.1, "banana": 0.9, "other_or_unknown": 0.0},
            }.items()
            for class_name, probability in values.items()
        ]
    ).to_parquet(probabilities_path, index=False)
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
    model_config = ModelConfig.model_validate(
        {
            "classifier": {
                "name": "test",
                "version": "test-v1",
                "input_size": 16,
                "unknown_class_policy": "retain_unknown_mass",
            },
            "calorie_model": {
                "distribution": "empirical",
                "small_class_policy": "exclude",
                "minimum_class_size": 2,
            },
        }
    )
    experiment = ExperimentConfig.model_validate(
        {
            "run_name": "test-run",
            "seed": 1,
            "interval_level": 0.9,
            "n_simulations": 100,
            "splits": {"train": 0.7, "validation": 0.15, "test": 0.15},
        }
    )

    compare_models(
        data_config,
        model_config,
        experiment,
        probabilities_path,
        predictions_path,
        metrics_path,
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(predictions_path)
    assert metrics["evaluation_split"] == "test"
    assert metrics["test_rows"] == 2
    assert metrics["bootstrap"]["method"] == "percentile cluster bootstrap over test source groups"
    assert metrics["models"]["M3"]["bootstrap_95_ci"]["mae_kcal"] is not None
    assert set(metrics["models"]) == {"M1", "M2", "M3"}
    assert "mean_crps_kcal" not in metrics["models"]["M1"]
    assert "mean_crps_kcal" in metrics["models"]["M2"]
    assert "mean_crps_kcal" in metrics["models"]["M3"]
    assert "M3_confidence_and_error" in metrics["diagnostics"]
    assert sum(metrics["diagnostics"]["M3_pit_histogram"]["counts"]) == 2
    assert len(predictions) == 6
    assert set(predictions.model_id) == {"M1", "M2", "M3"}
    assert set(predictions.sample_id) == {"test-a", "test-b"}
    assert set(predictions.group_id) == {"test-a", "test-b"}
    assert predictions.loc[predictions.model_id.eq("M1"), "interval_lower_kcal"].isna().all()
