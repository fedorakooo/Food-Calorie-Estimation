import json
from pathlib import Path

import pandas as pd

from food_calorie_estimation.config import DataConfig
from food_calorie_estimation.pipelines.generate_figures import generate_figures


def test_generate_figures_writes_complete_diagnostic_set(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.parquet"
    probabilities_path = tmp_path / "probabilities.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    metrics_path = tmp_path / "metrics.json"
    output_dir = tmp_path / "figures"
    pd.DataFrame(
        {
            "sample_id": ["train-a", "train-b", "valid-a", "valid-b", "test-a", "test-b"],
            "group_id": ["train-a", "train-b", "valid-a", "valid-b", "test-a", "test-b"],
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "is_eligible": [True] * 6,
            "food_class": ["apple", "banana", "apple", "banana", "apple", "banana"],
            "calories_kcal": [50.0, 90.0, 55.0, 95.0, 52.0, 92.0],
        }
    ).to_parquet(observations_path, index=False)
    pd.DataFrame(
        [
            {"sample_id": sample, "class_name": label, "calibrated_probability": probability}
            for sample, values in {
                "valid-a": {"apple": 0.8, "banana": 0.2},
                "valid-b": {"apple": 0.3, "banana": 0.7},
            }.items()
            for label, probability in values.items()
        ]
    ).to_parquet(probabilities_path, index=False)
    prediction_rows = []
    for model_id in ("M2", "M3"):
        for sample_id, actual, estimated, confidence, entropy in (
            ("test-a", 52.0, 56.0, 0.8, 0.5),
            ("test-b", 92.0, 87.0, 0.7, 0.6),
        ):
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "model_id": model_id,
                    "actual_calories_kcal": actual,
                    "estimated_calories_kcal": estimated,
                    "predicted_class_probability": confidence,
                    "prediction_entropy": entropy,
                    "interval_lower_kcal": estimated - 10,
                    "interval_upper_kcal": estimated + 10,
                }
            )
    pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "interval_level": 0.9,
                "models": {"M2": {"interval_coverage": 0.5}, "M3": {"interval_coverage": 1.0}},
                "diagnostics": {
                    "M2_pit_histogram": {"bin_edges": [index / 10 for index in range(11)], "counts": [0] * 10},
                    "M3_pit_histogram": {"bin_edges": [index / 10 for index in range(11)], "counts": [0] * 10},
                },
            }
        ),
        encoding="utf-8",
    )
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

    generated = generate_figures(data_config, probabilities_path, predictions_path, metrics_path, output_dir)

    assert len(generated) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)
