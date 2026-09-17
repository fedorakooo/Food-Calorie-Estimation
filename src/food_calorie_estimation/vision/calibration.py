from __future__ import annotations

import numpy as np

from food_calorie_estimation.vision.classifier import softmax


def negative_log_likelihood(logits: np.ndarray, target_indices: np.ndarray, temperature: float) -> float:
    probabilities = softmax(logits, temperature)
    target_probabilities = probabilities[np.arange(len(target_indices)), target_indices]
    return float(-np.log(np.clip(target_probabilities, 1e-12, 1)).mean())


def fit_temperature(logits: np.ndarray, target_indices: np.ndarray) -> float:
    """Choose a frozen temperature solely by validation negative log likelihood."""

    if len(logits) == 0:
        raise ValueError("validation logits cannot be empty")
    candidates = np.geomspace(0.05, 20.0, 401)
    losses = [negative_log_likelihood(logits, target_indices, float(value)) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def calibration_metrics(logits: np.ndarray, target_indices: np.ndarray, temperature: float) -> dict[str, float]:
    """Compute reproducible validation classification/calibration diagnostics."""

    probabilities = softmax(logits, temperature)
    one_hot = np.eye(probabilities.shape[1])[target_indices]
    return {
        "accuracy": float((probabilities.argmax(axis=1) == target_indices).mean()),
        "negative_log_likelihood": negative_log_likelihood(logits, target_indices, temperature),
        "brier_score": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
    }
