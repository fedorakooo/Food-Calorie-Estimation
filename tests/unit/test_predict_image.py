import json
from pathlib import Path

import numpy as np
from PIL import Image

from food_calorie_estimation.pipelines.predict_image import predict_image


def test_predict_image_loads_frozen_artifacts_and_returns_seeded_result(tmp_path: Path) -> None:
    image_path = tmp_path / "apple.png"
    Image.new("RGB", (24, 24), color=(255, 0, 0)).save(image_path)
    vision_path = tmp_path / "vision.json"
    calorie_path = tmp_path / "calorie.json"
    feature = np.zeros(48)
    feature[15] = feature[16] = feature[32] = 1 / 3
    vision_path.write_text(
        json.dumps(
            {
                "labels": ["apple", "banana"],
                "centroids": [feature.tolist(), np.roll(feature, 1).tolist()],
                "scale": 0.01,
                "input_size": 24,
                "temperature": 1.0,
                "model_version": "vision-test-v1",
            }
        ),
        encoding="utf-8",
    )
    calorie_path.write_text(
        json.dumps(
            {
                "model_version": "calorie-test-v1",
                "classes": {
                    "apple": {"values_kcal": [40.0, 60.0], "source_groups": 12},
                    "banana": {"values_kcal": [80.0, 100.0], "source_groups": 12},
                },
            }
        ),
        encoding="utf-8",
    )

    first = predict_image(image_path, vision_path, calorie_path, 1_000, 0.9, 7)
    second = predict_image(image_path, vision_path, calorie_path, 1_000, 0.9, 7)

    assert first == second
    assert first["classifier_version"] == "vision-test-v1"
    assert first["calorie_model_version"] == "calorie-test-v1"
    assert first["n_simulations"] == 1_000
    assert first["seed"] == 7
    assert first["supported_probability_mass"] == 1.0
    assert first["warnings"] == ["research_only", "portion_and_recipe_unobserved"]
    assert isinstance(first["top_classes"], list)
    assert len(first["top_classes"]) == 2
