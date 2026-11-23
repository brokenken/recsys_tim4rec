import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")

DATASETS = {
    "movielens": "MovieLens-1M",
    "steam": "Steam",
}

METRICS = {
    "hit@10": "Hit@10",
    "ndcg@10": "NDCG@10",
    "mrr@10": "MRR@10",
}


def load_results(dataset_key: str) -> dict:
    path = RESULTS_DIR / f"{dataset_key}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_comparison() -> Path:
    results = {key: load_results(key) for key in DATASETS}
    metric_keys = list(METRICS.keys())
    metric_labels = list(METRICS.values())
    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dataset_count = len(DATASETS)
    for i, (dataset_key, dataset_name) in enumerate(DATASETS.items()):
        values = [results[dataset_key]["test_result"][metric] for metric in metric_keys]
        offset = (i - (dataset_count - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=dataset_name)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title("Качество рекомендаций TiM4Rec")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 0.35)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True,)
    output_path = RESULTS_DIR / "metrics_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_comparison()
