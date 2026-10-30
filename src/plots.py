import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from flawed_data import ID_COL, TARGET

LEAKY_COLOR = "#d1495b"
FIXED_COLOR = "#00798c"
NEUTRAL_COLOR = "#8d99ae"
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LEAK_FEATURES = ("churn_risk_score", "exit_survey_score")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "font.size": 10,
    }
)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_class_balance(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df[TARGET].map({0: "Retained", 1: "Churned"}).value_counts()
    counts = counts.reindex(["Retained", "Churned"])
    bars = axes[0].bar(counts.index, counts.values, color=[FIXED_COLOR, LEAKY_COLOR])
    for bar, value in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 40, f"{value:,}", ha="center")
    axes[0].set_title("Class balance")
    axes[0].set_ylabel("Customers")

    rate = df.groupby("Contract")[TARGET].mean().mul(100).sort_values()
    bars = axes[1].bar(rate.index, rate.values, color=NEUTRAL_COLOR)
    for bar, value in zip(bars, rate.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%", ha="center")
    axes[1].set_title("Churn rate by contract type")
    axes[1].set_ylabel("Churn rate (%)")
    fig.tight_layout()
    return fig


def plot_leak_evidence(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for value, label, color in ((0, "Retained", FIXED_COLOR), (1, "Churned", LEAKY_COLOR)):
        axes[0].hist(
            df.loc[df[TARGET] == value, "churn_risk_score"],
            bins=25,
            alpha=0.7,
            color=color,
            label=label,
        )
    axes[0].set_title("churn_risk_score separates the classes")
    axes[0].set_xlabel("churn_risk_score")
    axes[0].set_ylabel("Rows")
    axes[0].legend()

    answered = df.groupby(TARGET)["exit_survey_score"].apply(lambda s: s.notna().mean() * 100)
    answered = answered.reindex([0, 1])
    bars = axes[1].bar(["Retained", "Churned"], answered.values, color=[FIXED_COLOR, LEAKY_COLOR])
    for bar, value in zip(bars, answered.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%", ha="center")
    axes[1].set_title("exit_survey_score exists only after churn")
    axes[1].set_ylabel("Rows with a survey score (%)")
    axes[1].set_ylim(0, 112)

    duplicates = int(df.duplicated(subset=ID_COL).sum())
    unique = int(df[ID_COL].nunique())
    bars = axes[2].bar(["Unique customers", "Duplicate rows"], [unique, duplicates], color=[FIXED_COLOR, LEAKY_COLOR])
    for bar, value in zip(bars, [unique, duplicates]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 80, f"{value:,}", ha="center")
    axes[2].set_ylim(0, unique * 1.12)
    axes[2].set_title("Duplicate customer rows")
    axes[2].set_ylabel("Rows")
    fig.tight_layout()
    return fig


def plot_metrics_comparison(leaky, fixed):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(METRIC_NAMES))
    width = 0.38
    leaky_values = [leaky["metrics"][m] for m in METRIC_NAMES]
    fixed_values = [fixed["metrics"][m] for m in METRIC_NAMES]
    bars_leaky = ax.bar(x - width / 2, leaky_values, width, color=LEAKY_COLOR, label="Leaky pipeline")
    bars_fixed = ax.bar(x + width / 2, fixed_values, width, color=FIXED_COLOR, label="Fixed pipeline")
    for bars in (bars_leaky, bars_fixed):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.12)
    ax.set_title("Leaky vs fixed pipeline on the held-out test set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_roc(leaky, fixed):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for result, color in ((leaky, LEAKY_COLOR), (fixed, FIXED_COLOR)):
        fpr, tpr, _ = roc_curve(result["y_test"], result["proba"])
        auc = result["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{result['name']} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=NEUTRAL_COLOR, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion(leaky, fixed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, result in zip(axes, (leaky, fixed)):
        matrix = confusion_matrix(result["y_test"], result["pred"])
        ax.imshow(matrix, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color=color, fontsize=13)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Retained", "Churned"])
        ax.set_yticklabels(["Retained", "Churned"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{result['name']} pipeline")
    fig.tight_layout()
    return fig


def plot_importance(result, top_n=10):
    order = np.argsort(result["importances"])[::-1][:top_n][::-1]
    names = result["features"][order]
    values = result["importances"][order]
    base = LEAKY_COLOR if result["name"] == "Leaky" else FIXED_COLOR
    colors = [LEAKY_COLOR if any(flag in n for flag in LEAK_FEATURES) else base for n in names]
    if result["name"] == "Leaky":
        colors = [LEAKY_COLOR if any(flag in n for flag in LEAK_FEATURES) else NEUTRAL_COLOR for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} features: {result['name'].lower()} pipeline")
    fig.tight_layout()
    return fig


def plot_ablation(table):
    labels = table["variant"].tolist()[::-1]
    values = table["roc_auc"].tolist()[::-1]
    colors = []
    for label in labels:
        if label == "Clean baseline":
            colors.append(FIXED_COLOR)
        elif label.startswith("All five"):
            colors.append(LEAKY_COLOR)
        else:
            colors.append(NEUTRAL_COLOR)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    ax.set_xlim(0.5, 1.05)
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Impact of each leakage source added to the clean pipeline")
    fig.tight_layout()
    return fig
