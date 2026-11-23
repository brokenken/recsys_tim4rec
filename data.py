import ast
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
import hashlib
from recbole.config import Config
from recbole.data import create_dataset


def make_user_token(value: str) -> str:
    return hashlib.blake2b(str(value).encode("utf-8"), digest_size=16).hexdigest()


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_available_datasets(config: dict) -> list[str]:
    return list(config["datasets"].keys())


def normalize_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        median_value = numeric.dropna().median()
        if median_value > 10_000_000_000:
            numeric = numeric / 1000.0
        return numeric
    datetime_values = pd.to_datetime(series, errors="coerce", utc=True)
    result = pd.Series(index=series.index, dtype="float64")
    valid = datetime_values.notna()
    result.loc[valid] = (datetime_values.loc[valid].astype("int64") / 1_000_000_000)
    return result


def load_standard_hf_dataset(dataset_config: dict) -> pd.DataFrame:
    hf_path = dataset_config["hf_path"]
    dataset_name = dataset_config["name"]
    print(f"Downloading {dataset_name}")
    print(f"Hugging Face: {hf_path}")
    dataset = load_dataset(hf_path)
    if isinstance(dataset, DatasetDict):
        frames = []
        for split_name, split in dataset.items():
            print(f"  split '{split_name}': \n {len(split):,}")
            frames.append(split.to_pandas())
        dataframe = pd.concat(frames, ignore_index=True)
    else:
        dataframe = dataset.to_pandas()
    print(f"Total downloaded: \n {len(dataframe):,}")
    return dataframe


def preprocess_standard_dataset(dataframe: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    columns = dataset_config["columns"]
    rename_map = {columns["user"]: "user_id", columns["item"]: "item_id", columns["timestamp"]: "timestamp"}
    if "rating" in columns:
        rename_map[columns["rating"]] = "rating"
    required_columns = [columns["user"], columns["item"], columns["timestamp"]]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError()
    dataframe = dataframe.rename(columns=rename_map)
    columns_to_keep = ["user_id", "item_id","timestamp"]
    if "rating" in dataframe.columns:
        columns_to_keep.append("rating")
    dataframe = dataframe[columns_to_keep].copy()
    dataframe = dataframe.dropna(subset=["user_id", "item_id", "timestamp"])
    dataframe["user_id"] = (dataframe["user_id"].astype(str))
    dataframe["item_id"] = (dataframe["item_id"].astype(str))
    dataframe["timestamp"] = (normalize_timestamp(dataframe["timestamp"]))
    dataframe = dataframe.dropna(subset=["timestamp"])
    dataframe = dataframe.sort_values(by=["user_id", "timestamp"])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def print_dataset_statistics(dataframe: pd.DataFrame, dataset_name: str) -> None:
    users = dataframe["user_id"].nunique()
    items = dataframe["item_id"].nunique()
    interactions = len(dataframe)
    sparsity = (1.0 - interactions / (users * items) if users and items else 0.0)
    avg_interactions = (interactions / users if users else 0.0)
    print("Dataset statistics")
    print("-" * 40)
    print(f"Dataset: {dataset_name}")
    print(f"Users: {users:,}")
    print(f"Items: {items:,}")
    print(f"Interactions: {interactions:,}")
    print(f"Average interactions/user: \n {avg_interactions:.2f}")
    print(f"Sparsity: {sparsity:.6f}")
    print("-" * 40)


def save_recbole_dataset(dataframe: pd.DataFrame, dataset_config: dict, data_dir: str) -> Path:
    recbole_name = dataset_config["recbole_name"]
    output_directory = (Path(data_dir) / recbole_name)
    output_directory.mkdir(parents=True, exist_ok=True,)
    output_path = (output_directory / f"{recbole_name}.inter")
    recbole_dataframe = dataframe[["user_id", "item_id","timestamp"]].copy()
    recbole_dataframe = (
        recbole_dataframe.rename(columns={"user_id": "user_id:token",
                                          "item_id": "item_id:token",
                                          "timestamp": "timestamp:float"}
                                 )
    )
    recbole_dataframe.to_csv(output_path,sep="\t", index=False)
    print(f"Saved: {output_path}")
    return output_path


def prepare_standard_dataset(dataset_config: dict, data_dir: str) -> Path:
    dataframe = load_standard_hf_dataset(dataset_config)
    dataframe = preprocess_standard_dataset(dataframe, dataset_config)
    print_dataset_statistics(dataframe, dataset_config["name"])
    return save_recbole_dataset(dataframe, dataset_config, data_dir)


def steam_date_to_timestamp(date_string: str, cache: dict) -> int:
    if date_string in cache:
        return cache[date_string]
    value = int(
        datetime.strptime(date_string,"%Y-%m-%d",).replace(tzinfo=timezone.utc).timestamp()
    )
    cache[date_string] = value
    return value


def prepare_steam_dataset(dataset_config: dict, data_dir: str) -> Path:
    repo_id = dataset_config["hf_path"]
    filename = dataset_config["filename"]
    recbole_name = dataset_config["recbole_name"]
    output_directory = (Path(data_dir) / recbole_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = (output_directory / f"{recbole_name}.inter")
    temp_path = output_path.with_suffix(".inter.tmp")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    interactions = 0
    skipped = 0
    date_cache = {}
    with (gzip.open(local_path, "rb") as source_file, open(temp_path, "w", encoding="utf-8", newline="")
                                            as output_file):
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["user_id:token", "item_id:token", "timestamp:float"])
        for line_number, raw_line in enumerate(source_file, start=1):
            try:
                line = raw_line.decode("utf-8", errors="replace",).strip()
            except Exception:
                skipped += 1
                continue
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                skipped += 1
                continue
            user_id = record.get("username")
            item_id = record.get("product_id")
            date = record.get("date")
            if user_id is None or item_id is None or not date:
                skipped += 1
                continue
            try:
                timestamp = (steam_date_to_timestamp(date, date_cache))
            except (ValueError, TypeError,):
                skipped += 1
                continue
            user_id = make_user_token(user_id)
            item_id = str(item_id)
            writer.writerow([str(user_id), str(item_id), timestamp])
            interactions += 1
            if (interactions % 250_000== 0):
                print(f"  processed: \n {interactions:,}")
    if interactions == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError()
    temp_path.replace(output_path)
    print("-" * 40)
    print(f"Interactions: \n {interactions:,}")
    print(f"Skipped records: \n {skipped:,}")
    print(f"Unique dates: \n {len(date_cache):,}")
    print("-" * 40)
    print(f"Saved: {output_path}")
    return output_path


def prepare_dataset(dataset_key: str, config: dict, force: bool = False) -> Path:
    datasets = config["datasets"]
    dataset_config = datasets[dataset_key]
    data_dir = config["paths"]["data_dir"]
    recbole_name = dataset_config["recbole_name"]
    expected_path = (Path(data_dir) / recbole_name / f"{recbole_name}.inter")
    if expected_path.exists() and not force:
        return expected_path
    source_type = dataset_config.get("source_type", "hf_dataset")
    if source_type == "hf_dataset":
        return prepare_standard_dataset(dataset_config, data_dir)
    if source_type == "steam_reviews":
        return prepare_steam_dataset(dataset_config, data_dir)


def inspect_recbole_dataset(dataset_key: str,project_config: dict):
    dataset_config = project_config["datasets"][dataset_key]
    recbole_name = dataset_config["recbole_name"]
    recbole_settings = dict(project_config["recbole"])
    recbole_settings.update(
        {
            "data_path": project_config["paths"]["data_dir"],
            "MAX_ITEM_LIST_LENGTH":dataset_config["max_seq_length"]
        }
    )
    config = Config(model="SASRec",dataset=recbole_name, config_dict=recbole_settings,)
    dataset = create_dataset(config)
    print("=" * 60)
    print(f"RecBole dataset: \n {dataset_config['name']}")
    print("=" * 60)
    print(dataset)
    return dataset
