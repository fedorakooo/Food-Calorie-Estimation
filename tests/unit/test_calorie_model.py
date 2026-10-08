import numpy as np
import pytest

from food_calorie_estimation.calorie_model import (
    empirical_crps,
    fit_empirical_distributions,
    mixture_moments,
    predict_hard_class_mean,
    prediction_warning_codes,
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
        simulate_mixture(np.array([[np.nan]]), ("apple",), {"apple": np.array([55.0])}, 10, 42)

    with pytest.raises(ValueError, match="supported class"):
        simulate_mixture(np.array([[1.0]]), ("other_or_unknown",), {}, 10, 42)


def test_summary_reports_expected_central_interval() -> None:
    summary = summarize_simulations(np.array([[10.0, 20.0, 30.0, 40.0]]), 0.5)[0]

    assert summary["estimated_calories_kcal"] == 25.0
    assert summary["p50_kcal"] == 25.0
    assert summary["interval_lower_kcal"] == 17.5
    assert summary["interval_upper_kcal"] == 32.5


def test_analytic_mixture_moments_match_known_distribution() -> None:
    probabilities = np.array([[0.25, 0.75]])
    distributions = {"apple": np.array([0.0, 2.0]), "banana": np.array([4.0])}

    mean, variance = mixture_moments(probabilities, ("apple", "banana"), distributions)

    assert np.allclose(mean, [3.25])
    assert np.allclose(variance, [1.9375])
    assert np.allclose(predict_hard_class_mean(probabilities, ("apple", "banana"), distributions), [4.0])


def test_empirical_crps_matches_two_point_example() -> None:
    score = empirical_crps(np.array([1.0]), np.array([[0.0, 2.0]]))

    assert np.allclose(score, [0.5])


def test_sample_key_makes_mixture_draws_independent_of_batch_position() -> None:
    distributions = {"apple": np.array([10.0, 20.0]), "banana": np.array([30.0, 40.0])}
    combined = simulate_mixture(
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        ("apple", "banana"),
        distributions,
        n_simulations=100,
        seed=42,
        row_keys=("first", "second"),
    )
    alone = simulate_mixture(
        np.array([[0.2, 0.8]]),
        ("apple", "banana"),
        distributions,
        n_simulations=100,
        seed=42,
        row_keys=("second",),
    )

    assert np.array_equal(combined[1], alone[0])


def test_prediction_warning_codes_report_low_confidence_and_discarded_mass() -> None:
    warnings = prediction_warning_codes(np.array([[0.4, 0.6], [0.5, 0.5]]), np.array([1.0, 0.8]))

    assert warnings[0] == ["research_only", "portion_and_recipe_unobserved"]
    assert warnings[1] == [
        "research_only",
        "portion_and_recipe_unobserved",
        "unsupported_class_probability_discarded",
    ]
