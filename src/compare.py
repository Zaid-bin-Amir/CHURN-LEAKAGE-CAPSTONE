import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ablation import run_ablation
from fixed_pipeline import run_fixed_pipeline
from flawed_data import build_flawed_dataset, load_raw
from leaky_pipeline import run_leaky_pipeline
from plots import (
    METRIC_NAMES,
    plot_ablation,
    plot_class_balance,
    plot_confusion,
    plot_importance,
    plot_leak_evidence,
    plot_metrics_comparison,
    plot_roc,
    save_fig,
)

ROOT = Path(__file__).resolve().parent.parent


def main(data_path=None):
    data_path = data_path or str(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    out_dir = ROOT / "outputs"
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    leaky = run_leaky_pipeline(data_path)
    fixed = run_fixed_pipeline(data_path)
    ablation = run_ablation(data_path)

    raw = load_raw(data_path)
    flawed = build_flawed_dataset(data_path)

    figures = {
        "01_class_balance.png": plot_class_balance(raw),
        "02_leakage_evidence.png": plot_leak_evidence(flawed),
        "03_metrics_comparison.png": plot_metrics_comparison(leaky, fixed),
        "04_roc_curves.png": plot_roc(leaky, fixed),
        "05_confusion_matrices.png": plot_confusion(leaky, fixed),
        "06_leaky_feature_importance.png": plot_importance(leaky),
        "07_fixed_feature_importance.png": plot_importance(fixed),
        "08_leakage_ablation.png": plot_ablation(ablation),
    }
    for name, fig in figures.items():
        save_fig(fig, chart_dir / name)
        plt.close(fig)

    comparison = pd.DataFrame(
        {
            "metric": METRIC_NAMES,
            "leaky": [leaky["metrics"][m] for m in METRIC_NAMES],
            "fixed": [fixed["metrics"][m] for m in METRIC_NAMES],
        }
    )
    comparison["gap"] = comparison["leaky"] - comparison["fixed"]
    comparison.round(4).to_csv(out_dir / "metrics_comparison.csv", index=False)
    ablation.round(4).to_csv(out_dir / "ablation.csv", index=False)

    summary = {
        "leaky_rows": leaky["rows"],
        "fixed_rows": fixed["rows"],
        "leaky_test_overlap": leaky["test_overlap"],
        "fixed_test_overlap": fixed["test_overlap"],
        "leaky_train_accuracy": leaky["train_accuracy"],
        "fixed_train_accuracy": fixed["train_accuracy"],
        "fixed_cv_roc_auc_mean": fixed["cv_roc_auc_mean"],
        "fixed_cv_roc_auc_std": fixed["cv_roc_auc_std"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(comparison.round(4).to_string(index=False))
    print()
    print(ablation.round(4).to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    print()
    print(f"Charts saved to: {chart_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
