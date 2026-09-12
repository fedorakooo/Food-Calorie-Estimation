from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    """Base configuration that rejects misspelled or undocumented fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitConfig(StrictConfig):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SplitConfig":
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("train, validation, and test fractions must sum to 1.0")
        return self


class DataConfig(StrictConfig):
    dataset_name: str = Field(min_length=1)
    raw_root: Path
    images_dir: Path
    annotations_dir: Path
    energy_density_path: Path
    processed_path: Path
    seed: int = Field(ge=0)
    minimum_class_size: int = Field(ge=2)


class ExperimentConfig(StrictConfig):
    run_name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0)
    interval_level: float = Field(gt=0, lt=1)
    n_simulations: int = Field(ge=1)
    splits: SplitConfig


class ClassifierSettings(StrictConfig):
    """Settings for the lightweight, reproducible image classifier baseline."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_size: int = Field(ge=16)
    unknown_class_policy: str = Field(pattern=r"^retain_unknown_mass$")


class CalorieModelSettings(StrictConfig):
    """Reserved validated settings for the next calorie-distribution stage."""

    distribution: str = Field(min_length=1)
    small_class_policy: str = Field(min_length=1)
    minimum_class_size: int = Field(ge=2)


class ModelConfig(StrictConfig):
    """Model configuration used by vision and later calorie-model pipelines."""

    classifier: ClassifierSettings
    calorie_model: CalorieModelSettings


ConfigType = TypeVar("ConfigType", bound=StrictConfig)


def load_yaml_config(path: str | Path, model: type[ConfigType]) -> ConfigType:
    """Read a YAML mapping and validate it against a configuration model."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        data: Any = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {config_path} must be a YAML mapping")
    return model.model_validate(data)
