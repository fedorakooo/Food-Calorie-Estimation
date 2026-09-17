import numpy as np

from food_calorie_estimation.vision.calibration import fit_temperature, negative_log_likelihood
from food_calorie_estimation.vision.classifier import UNKNOWN_CLASS, align_probabilities, softmax


def test_label_alignment_retains_unmapped_probability_mass() -> None:
    labels, probabilities = align_probabilities(
        ("apple", "unmapped_food", "banana"),
        np.array([[0.2, 0.3, 0.5]]),
        ("apple", "banana"),
    )

    assert labels == ("apple", "banana", UNKNOWN_CLASS)
    assert np.allclose(probabilities, [[0.2, 0.5, 0.3]])
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_temperature_fit_uses_positive_temperature_and_improves_validation_nll() -> None:
    logits = np.array([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    targets = np.array([0, 1, 1, 0])

    temperature = fit_temperature(logits, targets)

    assert temperature > 1
    assert negative_log_likelihood(logits, targets, temperature) < negative_log_likelihood(logits, targets, 1.0)


def test_softmax_rows_sum_to_one() -> None:
    probabilities = softmax(np.array([[1.0, 2.0], [1000.0, 1001.0]]))

    assert np.allclose(probabilities.sum(axis=1), 1)
