import argparse
import json
from pathlib import Path

from food_calorie_estimation.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    load_yaml_config,
)
from food_calorie_estimation.pipelines.compare_models import compare_models
from food_calorie_estimation.pipelines.estimate_calories import estimate_calories
from food_calorie_estimation.pipelines.evaluate_calories import evaluate_calories
from food_calorie_estimation.pipelines.generate_figures import generate_figures
from food_calorie_estimation.pipelines.predict_image import predict_image
from food_calorie_estimation.pipelines.prepare_data import prepare_data
from food_calorie_estimation.pipelines.train_vision import train_vision


def main() -> None:
    parser = argparse.ArgumentParser(description="Food calorie estimation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-config", help="validate a YAML config")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--kind", choices=("data", "experiment", "model"), required=True)
    prepare_parser = subparsers.add_parser("prepare-data", help="audit and prepare source data")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    prepare_parser.add_argument("--audit", type=Path, default=Path("artifacts/runs/data-audit.json"))
    prepare_parser.add_argument("--splits", type=Path, default=Path("artifacts/runs/split-assignments.csv"))
    vision_parser = subparsers.add_parser("fit-vision", help="fit the image classifier and validation-only calibration")
    vision_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    vision_parser.add_argument("--model", type=Path, default=Path("configs/model.yaml"))
    vision_parser.add_argument("--artifact", type=Path, default=Path("artifacts/models/vision-baseline.json"))
    calorie_parser = subparsers.add_parser("estimate-calories", help="estimate calories with Monte Carlo")
    calorie_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    calorie_parser.add_argument("--model", type=Path, default=Path("configs/model.yaml"))
    calorie_parser.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    calorie_parser.add_argument(
        "--probabilities", type=Path, default=Path("artifacts/runs/class-probabilities.parquet")
    )
    calorie_parser.add_argument("--output", type=Path, default=Path("artifacts/runs/calorie-estimates.parquet"))
    calorie_parser.add_argument("--artifact", type=Path, default=Path("artifacts/models/calorie-empirical.json"))
    evaluation_parser = subparsers.add_parser(
        "evaluate-calories", help="evaluate calorie estimates on the held-out test split"
    )
    evaluation_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    evaluation_parser.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    evaluation_parser.add_argument("--estimates", type=Path, default=Path("artifacts/runs/calorie-estimates.parquet"))
    evaluation_parser.add_argument("--output", type=Path, default=Path("artifacts/runs/calorie-test-metrics.json"))
    comparison_parser = subparsers.add_parser("compare-models", help="compare M1/M2/M3 on the held-out test split")
    comparison_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    comparison_parser.add_argument("--model", type=Path, default=Path("configs/model.yaml"))
    comparison_parser.add_argument("--experiment", type=Path, default=Path("configs/experiment.yaml"))
    comparison_parser.add_argument(
        "--probabilities", type=Path, default=Path("artifacts/runs/class-probabilities.parquet")
    )
    comparison_parser.add_argument(
        "--predictions", type=Path, default=Path("artifacts/runs/calorie-model-predictions.parquet")
    )
    comparison_parser.add_argument("--output", type=Path, default=Path("artifacts/runs/model-comparison.json"))
    prediction_parser = subparsers.add_parser("predict", help="predict calories for one image using frozen artifacts")
    prediction_parser.add_argument("image", type=Path)
    prediction_parser.add_argument(
        "--vision-artifact", type=Path, default=Path("artifacts/models/vision-baseline.json")
    )
    prediction_parser.add_argument(
        "--calorie-artifact", type=Path, default=Path("artifacts/models/calorie-empirical.json")
    )
    prediction_parser.add_argument("--simulations", type=int, default=10_000)
    prediction_parser.add_argument("--interval", type=float, default=0.90)
    prediction_parser.add_argument("--seed", type=int, default=42)
    prediction_parser.add_argument("--top-k", type=int, default=3)
    prediction_parser.add_argument("--output", type=Path)
    figures_parser = subparsers.add_parser("generate-figures", help="generate diagnostics from frozen artifacts")
    figures_parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"))
    figures_parser.add_argument(
        "--probabilities", type=Path, default=Path("artifacts/runs/class-probabilities.parquet")
    )
    figures_parser.add_argument(
        "--predictions", type=Path, default=Path("artifacts/runs/calorie-model-predictions.parquet")
    )
    figures_parser.add_argument("--metrics", type=Path, default=Path("artifacts/runs/model-comparison.json"))
    figures_parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    vision_parser.add_argument("--probabilities", type=Path, default=Path("artifacts/runs/class-probabilities.parquet"))
    vision_parser.add_argument("--metrics", type=Path, default=Path("artifacts/runs/vision-validation.json"))
    args = parser.parse_args()

    if args.command == "predict":
        prediction = predict_image(
            args.image,
            args.vision_artifact,
            args.calorie_artifact,
            args.simulations,
            args.interval,
            args.seed,
            args.top_k,
        )
        rendered = json.dumps(prediction, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(f"{rendered}\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "generate-figures":
        data_config = load_yaml_config(args.data, DataConfig)
        generated = generate_figures(
            data_config,
            args.probabilities,
            args.predictions,
            args.metrics,
            args.output_dir,
        )
        print(f"Generated {len(generated)} figures in {args.output_dir}")
        return

    if args.command == "validate-config":
        if args.kind == "data":
            load_yaml_config(args.path, DataConfig)
        elif args.kind == "experiment":
            load_yaml_config(args.path, ExperimentConfig)
        else:
            load_yaml_config(args.path, ModelConfig)
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
    if args.command == "evaluate-calories":
        experiment = load_yaml_config(args.experiment, ExperimentConfig)
        evaluate_calories(data_config, experiment, args.estimates, args.output)
        print(f"Held-out calorie metrics: {args.output}")
        return

    model_config = load_yaml_config(args.model, ModelConfig)
    if args.command == "compare-models":
        experiment = load_yaml_config(args.experiment, ExperimentConfig)
        compare_models(
            data_config,
            model_config,
            experiment,
            args.probabilities,
            args.predictions,
            args.output,
        )
        print(f"Frozen model comparison: {args.output}")
        print(f"Per-model test predictions: {args.predictions}")
        return
    if args.command == "estimate-calories":
        experiment = load_yaml_config(args.experiment, ExperimentConfig)
        estimate_calories(data_config, model_config, experiment, args.probabilities, args.output, args.artifact)
        print(f"Calorie estimates: {args.output}")
        return
    train_vision(data_config, model_config, args.artifact, args.probabilities, args.metrics)
    print(f"Vision artifact: {args.artifact}")
    print(f"Class probabilities: {args.probabilities}")
    print(f"Validation metrics: {args.metrics}")
