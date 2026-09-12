import argparse
from pathlib import Path

from food_calorie_estimation.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    load_yaml_config,
)
from food_calorie_estimation.pipelines.prepare_data import prepare_data
from food_calorie_estimation.pipelines.train_vision import train_vision
from food_calorie_estimation.pipelines.estimate_calories import estimate_calories


def main() -> None:
    parser = argparse.ArgumentParser(description="Food calorie estimation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-config", help="validate a YAML config")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--kind", choices=("data", "experiment", "model"), required=True)
    prepare_parser = subparsers.add_parser("prepare-data", help="audit and prepare source data")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    prepare_parser.add_argument(
        "--audit", type=Path, default=Path("artifacts/runs/data-audit.json")
    )
    prepare_parser.add_argument(
        "--splits", type=Path, default=Path("artifacts/runs/split-assignments.csv")
    )
    vision_parser = subparsers.add_parser(
        "fit-vision", help="fit the image classifier and validation-only calibration"
    )
    vision_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    vision_parser.add_argument("--model", type=Path, default=Path("configs/model.yaml"))
    vision_parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/models/vision-baseline.json")
    )
    calorie_parser = subparsers.add_parser("estimate-calories", help="estimate calories with Monte Carlo")
    calorie_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    calorie_parser.add_argument("--model", type=Path, default=Path("configs/model.yaml"))
    calorie_parser.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    calorie_parser.add_argument("--probabilities", type=Path, default=Path("artifacts/runs/class-probabilities.parquet"))
    calorie_parser.add_argument("--output", type=Path, default=Path("artifacts/runs/calorie-estimates.parquet"))
    calorie_parser.add_argument("--artifact", type=Path, default=Path("artifacts/models/calorie-empirical.json"))
    vision_parser.add_argument(
        "--probabilities", type=Path, default=Path("artifacts/runs/class-probabilities.parquet")
    )
    vision_parser.add_argument(
        "--metrics", type=Path, default=Path("artifacts/runs/vision-validation.json")
    )
    args = parser.parse_args()

    if args.command == "validate-config":
        model = {
            "data": DataConfig,
            "experiment": ExperimentConfig,
            "model": ModelConfig,
        }[args.kind]
        load_yaml_config(args.path, model)
        print(f"Valid {args.kind} configuration: {args.path}")
        return

    if args.command == "prepare-data":
        config = load_yaml_config(args.config, DataConfig)
        experiment = load_yaml_config(args.experiment, ExperimentConfig)
        prepare_data(config, experiment, args.audit, args.splits)
        print(f"Prepared observations: {config.processed_path}")
        print(f"Data audit: {args.audit}")
        print(f"Eligible split assignments: {args.splits}")
        return

    data_config = load_yaml_config(args.data, DataConfig)
    model_config = load_yaml_config(args.model, ModelConfig)
    if args.command == "estimate-calories":
        experiment = load_yaml_config(args.experiment, ExperimentConfig)
        estimate_calories(data_config, model_config, experiment, args.probabilities, args.output, args.artifact)
        print(f"Calorie estimates: {args.output}")
        return
    train_vision(data_config, model_config, args.artifact, args.probabilities, args.metrics)
    print(f"Vision artifact: {args.artifact}")
    print(f"Class probabilities: {args.probabilities}")
    print(f"Validation metrics: {args.metrics}")
