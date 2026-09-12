import numpy as np
import pytest

from food_calorie_estimation.calorie_model import (
    fit_empirical_distributions,
    simulate_mixture,
    summarize_simulations,
)


def test_fit_empirical_distributions_excludes_small_classes() -> None:
    fitted = fit_empirical_distributions(
        np.array([40.0, 50.0, 60.0, 70.0]),
        np.array(["apple", "apple", "banana", "banana"]),
        minimum_class_size=2,
    )

    assert set(fitted) == {"apple", "banana"}
    assert np.array_equal(fitted["apple"], [40.0, 50.0])


def test_mixture_renormalizes_mass_over_supported_classes() -> None:
    samples = simulate_mixture(
        np.array([[0.25, 0.75]]),
        ("apple", "other_or_unknown"),
        {"apple": np.array([55.0])},
        n_simulations=20,
        seed=42,
    )

    assert samples.shape == (1, 20)
    assert np.all(samples == 55.0)


def test_mixture_rejects_invalid_probability_rows() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        simulate_mixture(
            np.array([[np.nan]]), ("apple",), {"apple": np.array([55.0])}, 10, 42
        )

    with pytest.raises(ValueError, match="supported class"):
        simulate_mixture(
            np.array([[1.0]]), ("other_or_unknown",), {}, 10, 42
        )


def test_summary_reports_expected_central_interval() -> None:
    summary = summarize_simulations(np.array([[10.0, 20.0, 30.0, 40.0]]), 0.5)[0]

    assert summary["estimated_calories_kcal"] == 25.0
    assert summary["p50_kcal"] == 25.0
    assert summary["interval_lower_kcal"] == 17.5
    assert summary["interval_upper_kcal"] == 32.5
