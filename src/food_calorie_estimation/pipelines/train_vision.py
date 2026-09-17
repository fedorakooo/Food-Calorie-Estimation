from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from food_calorie_estimation.config import DataConfig, ModelConfig
from food_calorie_estimation.vision.calibration import calibration_metrics, fit_temperature
from food_calorie_estimation.vision.classifier import (
    CentroidClassifier,
    align_probabilities,
    softmax,
)


def _records(
    observations: pd.DataFrame,
    logits: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    labels: tuple[str, ...],
    model_version: str,
) -> pd.DataFrame:
    top_indices = calibrated.argmax(axis=1)
    sorted_probabilities = np.sort(calibrated, axis=1)
    entropy = -(calibrated * np.log(np.clip(calibrated, 1e-12, 1))).sum(axis=1)
    rows: list[dict[str, object]] = []
    for row_index, observation in enumerate(observations.itertuples(index=False)):
        for class_index, class_name in enumerate(labels):
            rows.append(
                {
                    "sample_id": observation.sample_id,
                    "split": observation.split,
                    "class_name": class_name,
                    "raw_logit": float(logits[row_index, class_index]),
                    "raw_probability": float(raw[row_index, class_index]),
                    "calibrated_probability": float(calibrated[row_index, class_index]),
                    "top_class": labels[top_indices[row_index]],
                    "top_probability": float(calibrated[row_index, top_indices[row_index]]),
                    "entropy": float(entropy[row_index]),
                    "margin_top1_top2": float(
                        sorted_probabilities[row_index, -1] - sorted_probabilities[row_index, -2]
                    ),
                    "model_version": model_version,
                }
            )
    return pd.DataFrame(rows)


def train_vision(
    data_config: DataConfig,
    model_config: ModelConfig,
    artifact_path: Path,
    probability_path: Path,
    metrics_path: Path,
) -> None:
    """Train on train rows, calibrate on validation rows, and infer all eligible rows."""

    observations = pd.read_parquet(data_config.processed_path)
    observations = observations.loc[observations.is_eligible].copy()
    train = observations.loc[observations.split == "train"]
    validation = observations.loc[observations.split == "validation"]
    if train.empty or validation.empty:
        raise ValueError("processed observations require non-empty train and validation splits")
    settings = model_config.classifier
    classifier = CentroidClassifier.fit(train.image_path.tolist(), train.food_class.tolist(), settings.input_size)
    label_index = {label: index for index, label in enumerate(classifier.labels)}
    validation_logits = classifier.logits_for_paths(validation.image_path.tolist())
    validation_targets = validation.food_class.map(label_index).to_numpy()
    temperature = fit_temperature(validation_logits, validation_targets)
    raw_metrics = calibration_metrics(validation_logits, validation_targets, 1.0)
    calibrated_metrics = calibration_metrics(validation_logits, validation_targets, temperature)
    all_logits = classifier.logits_for_paths(observations.image_path.tolist())
    raw = softmax(all_logits)
    calibrated = softmax(all_logits, temperature)
    # The fitted baseline has an identical source/canonical label space, but keep
    # alignment in the pipeline so a future external adapter cannot discard mass.
    labels, raw = align_probabilities(classifier.labels, raw, classifier.labels)
    _, calibrated = align_probabilities(classifier.labels, calibrated, classifier.labels)
    aligned_logits = np.column_stack([all_logits, np.full(len(all_logits), -np.inf)])
    records = _records(observations, aligned_logits, raw, calibrated, labels, settings.version)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "classifier_name": settings.name,
                "model_version": settings.version,
                "input_size": settings.input_size,
                "labels": classifier.labels,
                "unknown_class": labels[-1],
                "unknown_class_policy": settings.unknown_class_policy,
                "centroids": classifier.centroids.tolist(),
                "scale": classifier.scale,
                "temperature": temperature,
                "calibration_split": "validation",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(probability_path, index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "calibration_split": "validation",
                "temperature": temperature,
                "uncalibrated": raw_metrics,
                "calibrated": calibrated_metrics,
                "validation_rows": len(validation),
                "train_rows": len(train),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
