from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Global style settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def clean_axes(ax):
    """Remove top/right spines and make grid cleaner."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)


def annotate_bars(ax):
    """Add value labels on bars."""
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords='offset points')


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

evaluation_file = RESULTS_DIR / "least_to_most_results_evaluation.csv"
df = pd.read_csv(evaluation_file)

# keep only window-size rows
window_df = df[df["metric_group"] == "window_size"].copy()

# Accuracy line plot
accuracy_df = window_df[window_df["metric_name"] == "accuracy"].copy()
accuracy_df["group_value"] = accuracy_df["group_value"].astype(float).astype(int)
accuracy_df = accuracy_df.sort_values("group_value")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    accuracy_df["group_value"],
    accuracy_df["metric_value"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Window Size")
ax.set_ylabel("Accuracy")
ax.set_title("Least-to-Most Accuracy by Window Size")
ax.set_xticks([1, 3, 5, 7, 9, 11])
ax.set_ylim(0, 1)

clean_axes(ax)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "least_to_most_accuracy_by_window.png")



# Macro F1 line plot
f1_df = window_df[window_df["metric_name"] == "macro_f1"].copy()
f1_df["group_value"] = f1_df["group_value"].astype(float).astype(int)
f1_df = f1_df.sort_values("group_value")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    f1_df["group_value"],
    f1_df["metric_value"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Window Size")
ax.set_ylabel("Macro F1")
ax.set_title("Least-to-Most Macro F1 by Window Size")
ax.set_xticks([1, 3, 5, 7, 9, 11])
ax.set_ylim(0, 1)

clean_axes(ax)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "least_to_most_macro_f1_by_window.png")



# Grouped bar chart (window size)
window_pivot = window_df.pivot(
    index="group_value",
    columns="metric_name",
    values="metric_value"
).reset_index()

window_pivot["group_value"] = window_pivot["group_value"].astype(float).astype(int)
window_pivot = window_pivot.sort_values("group_value")

x = np.arange(len(window_pivot["group_value"]))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - 1.5 * width, window_pivot["accuracy"], width, label="Accuracy", alpha=0.9)
bars2 = ax.bar(x - 0.5 * width, window_pivot["macro_precision"], width, label="Macro Precision", alpha=0.9)
bars3 = ax.bar(x + 0.5 * width, window_pivot["macro_recall"], width, label="Macro Recall", alpha=0.9)
bars4 = ax.bar(x + 1.5 * width, window_pivot["macro_f1"], width, label="Macro F1", alpha=0.9)

ax.set_xlabel("Window Size")
ax.set_ylabel("Metric Value")
ax.set_title("Least-to-Most Metrics by Window Size")
ax.set_xticks(x)
ax.set_xticklabels(window_pivot["group_value"])
ax.set_ylim(0, 1)

clean_axes(ax)
ax.legend()

annotate_bars(ax)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "least_to_most_metrics_by_window_bar.png")


# Overall metrics bar chart
overall_df = df[df["metric_group"] == "overall"].copy()

metric_order = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
overall_df["metric_name"] = pd.Categorical(
    overall_df["metric_name"],
    categories=metric_order,
    ordered=True
)
overall_df = overall_df.sort_values("metric_name")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(overall_df["metric_name"], overall_df["metric_value"], alpha=0.9)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Least-to-Most Overall Metrics")
ax.set_ylim(0, 1)

clean_axes(ax)
annotate_bars(ax)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "least_to_most_overall_metrics_bar.png")


# Metrics by label
label_df = df[df["metric_group"] == "label"].copy()

label_pivot = label_df.pivot(
    index="group_value",
    columns="metric_name",
    values="metric_value"
).reset_index()

label_pivot = label_pivot.sort_values("group_value")

x = np.arange(len(label_pivot["group_value"]))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - 1.5 * width, label_pivot["accuracy"], width, label="Accuracy", alpha=0.9)
bars2 = ax.bar(x - 0.5 * width, label_pivot["macro_precision"], width, label="Macro Precision", alpha=0.9)
bars3 = ax.bar(x + 0.5 * width, label_pivot["macro_recall"], width, label="Macro Recall", alpha=0.9)
bars4 = ax.bar(x + 1.5 * width, label_pivot["macro_f1"], width, label="Macro F1", alpha=0.9)

ax.set_xlabel("Emotion Label")
ax.set_ylabel("Metric Value")
ax.set_title("Least-to-Most Metrics by Emotion Label")
ax.set_xticks(x)
ax.set_xticklabels(label_pivot["group_value"], rotation=45)
ax.set_ylim(0, 1)

clean_axes(ax)
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "least_to_most_metrics_by_label_bar.png")

plt.show()