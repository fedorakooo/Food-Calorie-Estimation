from pathlib import Path

import pandas as pd
import pytest

from food_calorie_estimation.config import SplitConfig
from food_calorie_estimation.data.ingest import load_energy_density, parse_ecustfd_filename
from food_calorie_estimation.data.split import assign_group_safe_splits


def test_parse_ecustfd_filename() -> None:
    assert parse_ecustfd_filename("fired_dough_twist001T(12).JPG") == (
        "fired_dough_twist001",
        "T",
        12,
    )


@pytest.mark.parametrize("filename", ["apple001.JPG", "apple001X(1).JPG", "apple001T(x).JPG"])
def test_invalid_ecustfd_filename_is_rejected(filename: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_ecustfd_filename(filename)


def test_energy_density_requires_source(tmp_path: Path) -> None:
    path = tmp_path / "density.yaml"
    path.write_text("classes:\n  apple:\n    kcal_per_100g: 52\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires a source"):
        load_energy_density(path)


def test_repository_energy_mapping_validates() -> None:
    root = Path(__file__).parents[2]
    densities = load_energy_density(root / "configs/energy_density.yaml")

    assert densities["apple"] == 62
    assert "bun" not in densities


def test_group_safe_split_keeps_each_group_together() -> None:
    observations = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
            "food_class": [
                "apple",
                "apple",
                "apple",
                "apple",
                "apple",
                "apple",
                "banana",
                "banana",
                "banana",
                "banana",
                "banana",
                "banana",
            ],
        }
    )
    splits = assign_group_safe_splits(
        observations, SplitConfig(train=0.5, validation=0.25, test=0.25), seed=7
    )

    assert splits.groupby(observations.group_id).nunique().eq(1).all()
    assert set(splits).issubset({"train", "validation", "test"})
