from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from food_calorie_estimation.calorie_model import (
    mixture_moments,
    normalize_supported_probabilities,
    prediction_warning_codes,
    simulate_mixture,
    summarize_simulations,
)
from food_calorie_estimation.vision.classifier import CentroidClassifier, softmax


def _load_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{artifact_name} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{artifact_name} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object")
    return value


def _load_classifier(path: Path) -> tuple[CentroidClassifier, float, str]:
    artifact = _load_json_object(path, "vision artifact")
    try:
        labels = tuple(artifact["labels"])
        centroids = np.asarray(artifact["centroids"], dtype=float)
        scale = float(artifact["scale"])
        input_size = int(artifact["input_size"])
        temperature = float(artifact["temperature"])
        model_version = str(artifact["model_version"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("vision artifact has invalid or missing model fields") from error
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("vision artifact labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("vision artifact labels must be unique")
    if centroids.ndim != 2 or centroids.shape[0] != len(labels) or centroids.shape[1] != 48:
        raise ValueError("vision artifact centroids must have shape (labels, 48)")
    if not np.all(np.isfinite(centroids)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("vision artifact centroids and scale must be finite with a positive scale")
    if input_size < 16 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("vision artifact input size and temperature are invalid")
    return CentroidClassifier(labels, centroids, scale, input_size), temperature, model_version


def _load_calorie_distributions(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int], str]:
    artifact = _load_json_object(path, "calorie artifact")
    raw_classes = artifact.get("classes")
    if not isinstance(raw_classes, dict) or not raw_classes:
        raise ValueError("calorie artifact must contain fitted classes")
    distributions: dict[str, np.ndarray] = {}
    source_groups: dict[str, int] = {}
    for label, description in raw_classes.items():
        if not isinstance(label, str) or not label or not isinstance(description, dict):
            raise ValueError("calorie artifact class entries are invalid")
        try:
            values = np.asarray(description["values_kcal"], dtype=float)
            group_count = int(description["source_groups"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"calorie artifact class {label!r} has invalid fields") from error
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"calorie artifact class {label!r} has invalid empirical values")
        if group_count < 1:
            raise ValueError(f"calorie artifact class {label!r} has invalid source-group support")
        distributions[label] = values
        source_groups[label] = group_count
    model_version = artifact.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("calorie artifact model_version is invalid")
    return distributions, source_groups, model_version


def predict_image(
    image_path: Path,
    vision_artifact_path: Path,
    calorie_artifact_path: Path,
    n_simulations: int,
    interval_level: float,
    seed: int,
    top_k: int = 3,
) -> dict[str, object]:
    """Load frozen artifacts and return a seeded calorie prediction for one image."""

    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be between 0 and 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not image_path.is_file():
        raise ValueError(f"input image does not exist: {image_path}")

    classifier, temperature, classifier_version = _load_classifier(vision_artifact_path)
    distributions, source_groups, calorie_model_version = _load_calorie_distributions(calorie_artifact_path)
    probabilities = softmax(classifier.logits_for_paths([str(image_path)]), temperature)
    supported = np.array([label in distributions for label in classifier.labels])
    supported_mass = (probabilities * supported).sum(axis=1)
    normalized = normalize_supported_probabilities(probabilities, classifier.labels, distributions)
    expected, variance = mixture_moments(probabilities, classifier.labels, distributions)
    image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = simulate_mixture(
        probabilities,
        classifier.labels,
        distributions,
        n_simulations,
        seed,
        row_keys=(image_digest,),
    )
    summary = summarize_simulations(samples, interval_level)[0]
    ranked_indices = np.argsort(probabilities[0])[::-1][: min(top_k, len(classifier.labels))]
    top_classes = [
        {
            "class": classifier.labels[index],
            "probability": float(probabilities[0, index]),
            "calorie_model_supported": bool(supported[index]),
        }
        for index in ranked_indices
    ]
    warnings = prediction_warning_codes(normalized, supported_mass)[0]
    top_supported_index = int(normalized[0].argmax())
    top_supported_label = classifier.labels[top_supported_index]
    if source_groups[top_supported_label] < 10:
        warnings.append("limited_independent_source_groups")

    return {
        "input_image": str(image_path),
        "input_sha256": image_digest,
        "expected_calories_kcal": float(expected[0]),
        "simulation_mean_calories_kcal": summary["estimated_calories_kcal"],
        "median_calories_kcal": summary["p50_kcal"],
        "std_calories_kcal": float(np.sqrt(variance[0])),
        "variance_calories_kcal2": float(variance[0]),
        "interval_level": interval_level,
        "prediction_interval_kcal": [summary["interval_lower_kcal"], summary["interval_upper_kcal"]],
        "p05_kcal": summary["p05_kcal"],
        "p95_kcal": summary["p95_kcal"],
        "n_simulations": n_simulations,
        "seed": seed,
        "supported_probability_mass": float(supported_mass[0]),
        "top_classes": top_classes,
        "classifier_version": classifier_version,
        "calorie_model_version": calorie_model_version,
        "top_supported_class_source_groups": source_groups[top_supported_label],
        "warnings": warnings,
    }
