from __future__ import annotations

import numpy as np
import pandas as pd

from food_calorie_estimation.config import SplitConfig

_SPLIT_NAMES = ("train", "validation", "test")


def assign_group_safe_splits(observations: pd.DataFrame, splits: SplitConfig, seed: int) -> pd.Series:
    """Assign every source group to one split while approximately preserving classes.

    A source group may contain multiple food classes (ECUSTFD mix images), so
    ordinary row-wise stratification would leak the same portion across splits.
    This deterministic greedy assignment instead balances each group's full
    class-count vector against the desired per-split class totals.
    """

    required = {"group_id", "food_class"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations missing split columns: {sorted(missing)}")
    if observations.empty:
        raise ValueError("cannot split an empty observation table")
    if observations.group_id.isna().any():
        raise ValueError("group_id cannot be null when splitting")

    group_class_counts = pd.crosstab(observations.group_id, observations.food_class)
    if len(group_class_counts) < len(_SPLIT_NAMES):
        raise ValueError("at least three source groups are required for train/validation/test splitting")

    fractions = np.array([splits.train, splits.validation, splits.test])
    target_class_counts = np.outer(fractions, group_class_counts.sum(axis=0).to_numpy())
    target_row_counts = fractions * len(observations)
    assigned_class_counts = np.zeros_like(target_class_counts, dtype=float)
    assigned_row_counts = np.zeros(len(_SPLIT_NAMES), dtype=float)
    assignments: dict[str, str] = {}

    generator = np.random.default_rng(seed)
    tie_breaker = dict(zip(group_class_counts.index, generator.random(len(group_class_counts)), strict=True))
    ordered_groups = sorted(
        group_class_counts.index,
        key=lambda group: (-int(group_class_counts.loc[[group]].to_numpy(dtype=float).sum()), tie_breaker[group]),
    )

    for group in ordered_groups:
        group_counts = group_class_counts.loc[[group]].to_numpy(dtype=float).ravel()
        group_rows = float(group_counts.sum())
        scores: list[float] = []
        for split_index in range(len(_SPLIT_NAMES)):
            projected_classes = assigned_class_counts.copy()
            projected_rows = assigned_row_counts.copy()
            projected_classes[split_index] += group_counts
            projected_rows[split_index] += group_rows
            class_error = np.abs(projected_classes - target_class_counts).sum()
            row_error = np.abs(projected_rows - target_row_counts).sum()
            scores.append(float(class_error + 0.1 * row_error))
        selected = int(np.argmin(scores))
        assigned_class_counts[selected] += group_counts
        assigned_row_counts[selected] += group_rows
        assignments[str(group)] = _SPLIT_NAMES[selected]

    return observations.group_id.astype(str).map(assignments).rename("split")
