from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "baseline-results"

evaluation_file = RESULTS_DIR / "gemini_zero_shot_results_evaluation.csv"
df = pd.read_csv(evaluation_file)


def add_line_labels(xvals, yvals, decimals=3):
    for x, y in zip(xvals, yvals):
        plt.text(
            x,
            y + 0.01,
            f"{y:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9
        )


def add_bar_labels(bars, decimals=3):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8
        )


# keep only window-size rows
window_df = df[df["metric_group"] == "window_size"].copy()

# accuracy data
accuracy_df = window_df[window_df["metric_name"] == "accuracy"].copy()
accuracy_df["group_value"] = accuracy_df["group_value"].astype(float).astype(int)
accuracy_df = accuracy_df.sort_values("group_value")

plt.figure(figsize=(8, 5))
plt.plot(accuracy_df["group_value"], accuracy_df["metric_value"], marker="o")
add_line_labels(accuracy_df["group_value"], accuracy_df["metric_value"])
plt.xlabel("Window Size")
plt.ylabel("Accuracy")
plt.title("Gemini Zero-Shot Accuracy by Window Size")
plt.xticks([1, 3, 5, 7, 9, 11])
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gemini_zero_shot_accuracy_by_window.png")
plt.show()

# macro f1 data
f1_df = window_df[window_df["metric_name"] == "macro_f1"].copy()
f1_df["group_value"] = f1_df["group_value"].astype(float).astype(int)
f1_df = f1_df.sort_values("group_value")

plt.figure(figsize=(8, 5))
plt.plot(f1_df["group_value"], f1_df["metric_value"], marker="o")
add_line_labels(f1_df["group_value"], f1_df["metric_value"])
plt.xlabel("Window Size")
plt.ylabel("Macro F1")
plt.title("Gemini Zero-Shot Macro F1 by Window Size")
plt.xticks([1, 3, 5, 7, 9, 11])
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gemini_zero_shot_macro_f1_by_window.png")
plt.show()

# grouped bar chart for all metrics by window size
window_pivot = window_df.pivot(
    index="group_value",
    columns="metric_name",
    values="metric_value"
).reset_index()

window_pivot["group_value"] = window_pivot["group_value"].astype(float).astype(int)
window_pivot = window_pivot.sort_values("group_value")

x = np.arange(len(window_pivot["group_value"]))
width = 0.2

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - 1.5 * width, window_pivot["accuracy"], width, label="Accuracy")
bars2 = plt.bar(x - 0.5 * width, window_pivot["macro_precision"], width, label="Macro Precision")
bars3 = plt.bar(x + 0.5 * width, window_pivot["macro_recall"], width, label="Macro Recall")
bars4 = plt.bar(x + 1.5 * width, window_pivot["macro_f1"], width, label="Macro F1")

add_bar_labels(bars1)
add_bar_labels(bars2)
add_bar_labels(bars3)
add_bar_labels(bars4)

plt.xlabel("Window Size")
plt.ylabel("Metric Value")
plt.title("Gemini Zero-Shot Metrics by Window Size")
plt.xticks(x, window_pivot["group_value"])
plt.ylim(0, 1)
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gemini_zero_shot_metrics_by_window_bar.png")
plt.show()

# overall metrics bar chart
overall_df = df[df["metric_group"] == "overall"].copy()

metric_order = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
overall_df["metric_name"] = pd.Categorical(overall_df["metric_name"], categories=metric_order, ordered=True)
overall_df = overall_df.sort_values("metric_name")

plt.figure(figsize=(8, 5))
bars = plt.bar(overall_df["metric_name"], overall_df["metric_value"])
add_bar_labels(bars)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Gemini Zero-Shot Overall Metrics")
plt.ylim(0, 1)
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gemini_zero_shot_overall_metrics_bar.png")
plt.show()

# grouped bar chart for all metrics by label
label_df = df[df["metric_group"] == "label"].copy()

label_pivot = label_df.pivot(
    index="group_value",
    columns="metric_name",
    values="metric_value"
).reset_index()

label_pivot = label_pivot.sort_values("group_value")

x = np.arange(len(label_pivot["group_value"]))
width = 0.2

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - 1.5 * width, label_pivot["accuracy"], width, label="Accuracy")
bars2 = plt.bar(x - 0.5 * width, label_pivot["macro_precision"], width, label="Macro Precision")
bars3 = plt.bar(x + 0.5 * width, label_pivot["macro_recall"], width, label="Macro Recall")
bars4 = plt.bar(x + 1.5 * width, label_pivot["macro_f1"], width, label="Macro F1")

add_bar_labels(bars1)
add_bar_labels(bars2)
add_bar_labels(bars3)
add_bar_labels(bars4)

plt.xlabel("Emotion Label")
plt.ylabel("Metric Value")
plt.title("Gemini Zero-Shot Metrics by Emotion Label")
plt.xticks(x, label_pivot["group_value"], rotation=45)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "gemini_zero_shot_metrics_by_label_bar.png")
plt.show()