from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as element_tree
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError

from food_calorie_estimation.config import DataConfig

_FILENAME_PATTERN = re.compile(
    r"^(?P<portion_id>.+?)(?P<view>[TS])\((?P<frame>\d+)\)\.(?P<extension>jpe?g)$",
    re.IGNORECASE,
)
_NON_FOOD_LABELS = frozenset({"coin"})


@dataclass
class AuditReport:
    """Counts and exceptions from one raw-data preparation run."""

    annotations_seen: int = 0
    food_objects_seen: int = 0
    eligible_rows: int = 0
    exclusion_counts: Counter[str] = field(default_factory=Counter)
    class_counts: Counter[str] = field(default_factory=Counter)
    split_counts: Counter[str] = field(default_factory=Counter)
    missing_images: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    duplicate_image_hashes: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible audit details with deterministic ordering."""

        return {
            "annotations_seen": self.annotations_seen,
            "food_objects_seen": self.food_objects_seen,
            "eligible_rows": self.eligible_rows,
            "exclusion_counts": dict(sorted(self.exclusion_counts.items())),
            "class_counts": dict(sorted(self.class_counts.items())),
            "split_counts": dict(sorted(self.split_counts.items())),
            "missing_images": self.missing_images,
            "unreadable_images": self.unreadable_images,
            "duplicate_image_hashes": self.duplicate_image_hashes,
        }


def parse_ecustfd_filename(filename: str) -> tuple[str, str, int]:
    """Return source portion ID, view, and frame from an ECUSTFD filename."""

    match = _FILENAME_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported ECUSTFD filename: {filename}")
    return match["portion_id"], match["view"].upper(), int(match["frame"])


def load_energy_density(path: Path) -> dict[str, float]:
    """Load only reviewed positive kcal-per-100-g entries from a YAML mapping."""

    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    classes = data.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("energy-density classes must be a YAML mapping")

    densities: dict[str, float] = {}
    for food_class, entry in classes.items():
        if not isinstance(entry, dict):
            raise ValueError(f"energy density for {food_class!r} must be a mapping")
        value = entry.get("kcal_per_100g")
        if value is None:
            continue
        density = float(value)
        if density <= 0:
            raise ValueError(f"energy density for {food_class!r} must be positive")
        if not entry.get("source"):
            raise ValueError(f"energy density for {food_class!r} requires a source")
        densities[str(food_class)] = density
    return densities


def load_mass_records(workbook: Path) -> dict[tuple[str, str], float]:
    """Load ECUSTFD's portion/class mass records keyed by source ID and label."""

    records: dict[tuple[str, str], float] = {}
    for sheet_name in pd.ExcelFile(workbook).sheet_names:
        sheet = pd.read_excel(workbook, sheet_name=sheet_name)
        expected = {"id", "type", "weight(g)"}
        if not expected.issubset(sheet.columns):
            raise ValueError(f"{sheet_name} is missing required mass columns")
        for row in sheet.itertuples(index=False):
            portion_id = str(getattr(row, "id")).strip()
            food_class = str(getattr(row, "type")).strip()
            mass_g = float(getattr(row, "_3"))
            if mass_g <= 0:
                raise ValueError(f"Non-positive mass for {portion_id}/{food_class}")
            key = (portion_id, food_class)
            if key in records:
                raise ValueError(f"Duplicate mass record for {portion_id}/{food_class}")
            records[key] = mass_g
    return records


def _image_metadata(image_path: Path) -> tuple[int, int, str]:
    """Verify an image can be decoded, then return its dimensions and hash."""

    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return width, height, digest


def _food_labels(annotation_path: Path) -> list[str]:
    root = element_tree.parse(annotation_path).getroot()
    return [
        name.text.strip()
        for name in root.findall("object/name")
        if name.text and name.text.strip() not in _NON_FOOD_LABELS
    ]


def build_observations(config: DataConfig) -> tuple[pd.DataFrame, AuditReport]:
    """Build one row per source image/food item without hiding validation errors."""

    raw_root = config.raw_root
    images_root = raw_root / config.images_dir
    annotations_root = raw_root / config.annotations_dir
    mass_records = load_mass_records(raw_root / "density.xls")
    densities = load_energy_density(config.energy_density_path)
    report = AuditReport()
    rows: list[dict[str, Any]] = []
    image_hash_paths: dict[str, list[str]] = {}

    for annotation_path in sorted(annotations_root.glob("*.xml")):
        report.annotations_seen += 1
        try:
            filename = element_tree.parse(annotation_path).findtext("filename")
            if not filename:
                raise ValueError("annotation_missing_filename")
            portion_id, view, frame = parse_ecustfd_filename(filename)
        except (element_tree.ParseError, ValueError) as error:
            report.exclusion_counts[f"annotation_error:{error}"] += 1
            continue

        image_path = images_root / filename
        labels = _food_labels(annotation_path)
        report.food_objects_seen += len(labels)
        if not image_path.is_file():
            report.missing_images.append(filename)
            report.exclusion_counts["missing_image"] += len(labels)
            continue
        try:
            width, height, image_hash = _image_metadata(image_path)
        except (OSError, UnidentifiedImageError):
            report.unreadable_images.append(filename)
            report.exclusion_counts["unreadable_image"] += len(labels)
            continue
        image_hash_paths.setdefault(image_hash, []).append(filename)

        for object_index, raw_label in enumerate(labels):
            food_class = raw_label.lower().replace(" ", "_")
            mass_g = mass_records.get((portion_id, food_class))
            density = densities.get(food_class)
            exclusion_reason: str | None = None
            if mass_g is None:
                exclusion_reason = "missing_mass_record"
            elif density is None:
                exclusion_reason = "missing_energy_density"
            calories = None if exclusion_reason else mass_g * density / 100
            eligible = exclusion_reason is None
            if eligible:
                report.eligible_rows += 1
            else:
                report.exclusion_counts[exclusion_reason] += 1
            report.class_counts[food_class] += 1
            rows.append(
                {
                    "sample_id": f"{Path(filename).stem}:food-{object_index}",
                    "image_path": str(image_path),
                    "source_dataset": "ecustfd",
                    "raw_label": raw_label,
                    "food_class": food_class,
                    "calories_kcal": calories,
                    "portion_mass_g": mass_g,
                    "calories_per_100g": density,
                    "split": None,
                    "group_id": portion_id,
                    "label_quality": "derived" if eligible else "unknown",
                    "is_eligible": eligible,
                    "exclusion_reason": exclusion_reason,
                    "view": view,
                    "frame": frame,
                    "image_width": width,
                    "image_height": height,
                    "image_sha256": image_hash,
                }
            )
    report.duplicate_image_hashes = {
        image_hash: paths for image_hash, paths in image_hash_paths.items() if len(paths) > 1
    }
    return pd.DataFrame(rows), report
