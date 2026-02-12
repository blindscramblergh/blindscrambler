import argparse
import glob
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(csv_path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    required_cols = {"Metric", "Training", "Test"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns {required_cols}: {csv_path}")

    metrics: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        metric_name = str(row["Metric"]).strip()
        metrics[metric_name] = {
            "Training": float(row["Training"]),
            "Test": float(row["Test"]),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multiclass metrics and plot boxplots"
    )
    parser.add_argument(
        "keyword",
        type=str,
        help="Unique keyword used in metrics CSV file names",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory containing metrics CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory to save the boxplot image",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.logs_dir, f"metrics_*.csv")
    matched_files = sorted(glob.glob(pattern))

    if not matched_files:
        raise SystemExit(
            f"No metrics files found for keyword '{args.keyword}' in {args.logs_dir}"
        )

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    train_values: Dict[str, List[float]] = {m: [] for m in metrics}
    test_values: Dict[str, List[float]] = {m: [] for m in metrics}

    for csv_path in matched_files:
        data = load_metrics(csv_path)
        for metric in metrics:
            if metric not in data:
                raise ValueError(f"Missing metric '{metric}' in {csv_path}")
            train_values[metric].append(data[metric]["Training"])
            test_values[metric].append(data[metric]["Test"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.boxplot(
            [train_values[metric], test_values[metric]],
            labels=["Training", "Test"],
        )
        ax.set_title(metric)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Metrics Boxplots for keyword '{args.keyword}'")
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, f"boxplot_{args.keyword}_{timestamp}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
