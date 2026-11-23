from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from data import prepare_dataset, inspect_recbole_dataset
from train import run_experiment, evaluate_saved_model


CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--force", action="store_true",)
    parser.add_argument("--inspect", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--evaluate-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config()
    available_datasets = list(config["datasets"].keys())
    if args.dataset == "all":
        selected_datasets = available_datasets
    else:
        if args.dataset not in available_datasets:
            raise ValueError(
                f"Unknown dataset '{args.dataset}'. "
                f"Available: {available_datasets}"
            )
        selected_datasets = [args.dataset]
    prepared_datasets = {}
    for dataset_key in selected_datasets:
        print()
        print("=" * 70)
        print(f"Dataset: {dataset_key}")
        print("=" * 70)
        path = prepare_dataset(dataset_key, config, force=args.force)
        prepared_datasets[dataset_key] = path
        if args.inspect:
            inspect_recbole_dataset(dataset_key, config)
        if args.train:
            run_experiment(dataset_key, config)
        elif args.evaluate_only:
            evaluate_saved_model(dataset_key, config)
    print()
    print("=" * 70)
    print("Prepared datasets")
    print("=" * 70)
    for key, path in prepared_datasets.items():
        print(f"{key:<12} -> {path}")


if __name__ == "__main__":
    main()