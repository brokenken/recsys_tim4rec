from __future__ import annotations
import gc
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed
from tim4rec import TiM4Rec


CHECKPOINT_ROOT = Path("saved")
RESULTS_ROOT = Path("results")


def merge_dict(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    result.update(deepcopy(override))
    return result


def get_checkpoint_path(dataset_key: str) -> Path:
    return (CHECKPOINT_ROOT / dataset_key / "tim4rec_best.pth")


def get_results_path(dataset_key: str) -> Path:
    return RESULTS_ROOT / f"{dataset_key}.json"


def build_recbole_config(dataset_key: str, project_config: dict) -> Config:
    dataset_cfg = project_config["datasets"][dataset_key]
    config_dict = deepcopy(project_config["recbole"])
    config_dict.update(deepcopy(project_config.get("model", {})))
    config_dict.update(deepcopy(project_config.get("training", {})))
    config_dict.update(deepcopy(dataset_cfg.get("model", {})))
    config_dict["data_path"] = project_config["paths"]["data_dir"]
    config_dict["MAX_ITEM_LIST_LENGTH"] = dataset_cfg["max_seq_length"]
    checkpoint_dir = CHECKPOINT_ROOT / dataset_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dict["checkpoint_dir"] = str(checkpoint_dir)
    config_dict.pop("checkpoint_path", None)
    return Config(model=TiM4Rec, dataset=dataset_cfg["recbole_name"], config_dict=config_dict)


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_results(dataset_key: str, project_config: dict, test_result: dict, checkpoint_path: Path,
                 best_valid_score=None, best_valid_result=None) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = get_results_path(dataset_key)
    if result_path.exists():
        try:
            with result_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    else:
        payload = {}
    old_test_result = payload.get("test_result", {})
    old_test_result.update(_to_plain(test_result))
    payload.update(
        {
            "dataset_key": dataset_key,
            "dataset_name": project_config["datasets"][dataset_key]["name"],
            "model": "TiM4Rec",
            "checkpoint": str(checkpoint_path),
            "last_evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "metrics": project_config["recbole"]["metrics"],
            "topk": project_config["recbole"]["topk"],
            "test_result": old_test_result
        }
    )
    if best_valid_score is not None:
        payload["best_validation_score"] = _to_plain(best_valid_score)
    if best_valid_result is not None:
        payload["best_validation_result"] = _to_plain(best_valid_result)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return result_path


def print_results(dataset_name: str, test_result: dict, best_valid_score=None, best_valid_result=None) -> None:
    print()
    print("=" * 70)
    print(f"Results: {dataset_name}")
    print("=" * 70)
    if best_valid_score is not None:
        print(f"Best validation score: {float(best_valid_score):.6f}")
    if best_valid_result is not None:
        print(f"Best validation result: {best_valid_result}")
    print("Test result:")
    for key, value in test_result.items():
        try:
            print(f"  {key:<12}: {float(value):.6f}")
        except (TypeError, ValueError):
            print(f"  {key:<12}: {value}")


def validate_checkpoint_config(checkpoint_path: Path,config: Config) -> None:
    checkpoint = torch.load(checkpoint_path)
    old_config = checkpoint.get("config")
    if old_config is None:
        return
    critical_keys = [
        "dataset",
        "hidden_size",
        "num_layers",
        "d_state",
        "d_conv",
        "expand",
        "head_dim",
        "chunk_size",
        "is_ffn",
        "is_time",
        "p2p_residual",
        "MAX_ITEM_LIST_LENGTH",
    ]
    mismatches = []
    for key in critical_keys:
        try:
            old_value = old_config[key]
            new_value = config[key]
        except Exception:
            continue
        if old_value != new_value:
            mismatches.append(f"{key}: checkpoint={old_value!r}, current={new_value!r}")


def run_experiment(dataset_key: str, project_config: dict) -> dict:
    dataset_cfg = project_config["datasets"][dataset_key]
    dataset_name = dataset_cfg["name"]
    checkpoint_path = get_checkpoint_path(dataset_key)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 70)
    print(f"Training TiM4Rec on {dataset_name}")
    print("=" * 70)
    config = build_recbole_config(dataset_key, project_config)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config,dataset,)
    init_seed(config["seed"], config["reproducibility"],)
    model = TiM4Rec(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    trainer.saved_model_file = str(checkpoint_path)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True,
                                                      show_progress=config["show_progress"])
    test_result = trainer.evaluate(test_data, load_best_model=True, model_file=str(checkpoint_path),
                                   show_progress=config["show_progress"],)
    print_results(dataset_name, test_result, best_valid_score, best_valid_result,)
    result_path = save_results(
        dataset_key=dataset_key,
        project_config=project_config,
        test_result=test_result,
        checkpoint_path=checkpoint_path,
        best_valid_score=best_valid_score,
        best_valid_result=best_valid_result,
    )
    result = {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "checkpoint": str(checkpoint_path),
        "results_file": str(result_path),
    }
    del trainer
    del model
    del train_data
    del valid_data
    del test_data
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_saved_model(dataset_key: str, project_config: dict) -> dict:
    dataset_cfg = project_config["datasets"][dataset_key]
    dataset_name = dataset_cfg["name"]
    checkpoint_path = get_checkpoint_path(dataset_key)
    config = build_recbole_config(dataset_key, project_config)
    validate_checkpoint_config(checkpoint_path,config)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = TiM4Rec(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    trainer.eval_collector.data_collect(train_data)
    test_result = trainer.evaluate(test_data, load_best_model=True, model_file=str(checkpoint_path),
                                   show_progress=config["show_progress"],)
    print_results(dataset_name, test_result,)
    result_path = save_results(
        dataset_key=dataset_key,
        project_config=project_config,
        test_result=test_result,
        checkpoint_path=checkpoint_path,
    )
    result = {
        "test_result": test_result,
        "checkpoint": str(checkpoint_path),
        "results_file": str(result_path),
    }
    del trainer
    del model
    del train_data
    del valid_data
    del test_data
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
