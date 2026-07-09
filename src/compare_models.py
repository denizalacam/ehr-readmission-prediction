from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


def load_metrics(path: Path) -> dict:
    metrics = {}

    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                name, value = line.strip().split(":", 1)

                try:
                    metrics[name] = float(value)
                except ValueError:
                    continue

    return metrics


def get_best_model(row):
    values = {
        "Logistic Regression": row["Logistic Regression"],
        "PyTorch MLP": row["PyTorch MLP"],
        "XGBoost": row["XGBoost"],
    }

    return max(values, key=values.get)


def main():
    baseline_metrics = load_metrics(RESULTS_DIR / "metrics.txt")
    pytorch_metrics = load_metrics(RESULTS_DIR / "pytorch_metrics.txt")
    xgboost_metrics = load_metrics(RESULTS_DIR / "xgboost_metrics.txt")

    comparison = pd.DataFrame(
        {
            "Metric": baseline_metrics.keys(),
            "Logistic Regression": baseline_metrics.values(),
            "PyTorch MLP": [
                pytorch_metrics[metric]
                for metric in baseline_metrics.keys()
            ],
            "XGBoost": [
                xgboost_metrics[metric]
                for metric in baseline_metrics.keys()
            ],
        }
    )

    comparison["Best Model"] = comparison.apply(
        get_best_model,
        axis=1,
    )

    output_path = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\nModel comparison saved:")
    print(output_path)

    print("\nComparison:")
    print(comparison)


if __name__ == "__main__":
    main()