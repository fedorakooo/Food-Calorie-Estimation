from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

UNKNOWN_CLASS = "other_or_unknown"


def image_feature(path: str | Path, input_size: int) -> np.ndarray:
    """Decode, orient, resize, and return a normalized RGB histogram feature."""

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").resize((input_size, input_size))
        pixels = np.asarray(image, dtype=np.uint8)
    histograms = [
        np.histogram(pixels[..., channel], bins=16, range=(0, 256))[0] for channel in range(3)
    ]
    feature = np.concatenate(histograms).astype(float)
    return feature / feature.sum()


@dataclass(frozen=True)
class CentroidClassifier:
    """Nearest-centroid classifier that exposes class logits rather than labels only."""

    labels: tuple[str, ...]
    centroids: np.ndarray
    scale: float
    input_size: int

    @classmethod
    def fit(cls, paths: list[str], labels: list[str], input_size: int) -> CentroidClassifier:
        classes = tuple(sorted(set(labels)))
        features = np.vstack([image_feature(path, input_size) for path in paths])
        label_array = np.asarray(labels)
        centroids = np.vstack([features[label_array == name].mean(axis=0) for name in classes])
        assigned_centroids = centroids[[classes.index(name) for name in labels]]
        distances = ((features - assigned_centroids) ** 2).sum(axis=1)
        # A positive, train-derived scale prevents arbitrary confidence magnitudes.
        scale = float(max(np.median(distances), 1e-8))
        return cls(classes, centroids, scale, input_size)

    def logits_for_paths(self, paths: list[str]) -> np.ndarray:
        features = np.vstack([image_feature(path, self.input_size) for path in paths])
        squared_distance = ((features[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return -squared_distance / self.scale


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return numerically stable probabilities for a positive temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def align_probabilities(
    source_labels: tuple[str, ...], probabilities: np.ndarray, canonical_labels: tuple[str, ...]
) -> tuple[tuple[str, ...], np.ndarray]:
    """Map source labels to canonical labels and retain unsupported mass explicitly."""

    if probabilities.shape[-1] != len(source_labels):
        raise ValueError("probability columns do not match source labels")
    labels = (*canonical_labels, UNKNOWN_CLASS)
    aligned = np.zeros((*probabilities.shape[:-1], len(labels)), dtype=float)
    positions = {label: index for index, label in enumerate(canonical_labels)}
    for source_index, source_label in enumerate(source_labels):
        destination = positions.get(source_label, len(canonical_labels))
        aligned[..., destination] += probabilities[..., source_index]
    return labels, aligned
