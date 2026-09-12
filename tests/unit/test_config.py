from pathlib import Path

import pytest
from pydantic import ValidationError

from food_calorie_estimation.config import DataConfig, ExperimentConfig, load_yaml_config


def test_repository_configs_validate() -> None:
    root = Path(__file__).parents[2]
    data = load_yaml_config(root / "configs/data.yaml", DataConfig)
    experiment = load_yaml_config(root / "configs/experiment.yaml", ExperimentConfig)

    assert data.dataset_name == "ecustfd"
    assert experiment.n_simulations == 10_000


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        ExperimentConfig.model_validate(
            {
                "run_name": "test-run",
                "seed": 1,
                "interval_level": 0.9,
                "n_simulations": 100,
                "splits": {"train": 0.7, "validation": 0.2, "test": 0.2},
            }
        )


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("dataset_name: ecustfd\nunknown: value\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="unknown"):
        load_yaml_config(config, DataConfig)
