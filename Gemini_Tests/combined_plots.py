from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR

files = [
    {
        "name": "zero_shot",
        "path": ROOT_DIR / "gemini-zeroshot" / "results" / "gemini_zero_shot_results_evaluation.csv"
    },
    {
        "name": "two_shot",
        "path": ROOT_DIR / "gemini_2shot" / "results" / "gemini_meld_emotion_results_evaluation.csv"
    },
    {
        "name": "least_to_most",
        "path": ROOT_DIR / "least_to_most" / "results" / "least_to_most_results_evaluation.csv"
    }
]

all_dfs = []

for file_info in files:
    df = pd.read_csv(file_info["path"])
    df["prompt_type"] = file_info["name"]
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Accuracy by window size
window_accuracy_df = combined_df[
    (combined_df["metric_group"] == "window_size") &
    (combined_df["metric_name"] == "accuracy")
].copy()

window_accuracy_df["group_value"] = pd.to_numeric(window_accuracy_df["group_value"]).astype(int)
window_accuracy_df = window_accuracy_df.sort_values(["prompt_type", "group_value"])

plt.figure(figsize=(9, 6))
for prompt_type in window_accuracy_df["prompt_type"].unique():
    subset = window_accuracy_df[window_accuracy_df["prompt_type"] == prompt_type]
    plt.plot(subset["group_value"], subset["metric_value"], marker="o", label=prompt_type)

plt.xlabel("Window Size")
plt.ylabel("Accuracy")
plt.title("Accuracy by Window Size Across Prompting Methods")
plt.xticks([1, 3, 5, 7, 9, 11])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT_DIR / "combined_accuracy_by_window.png")
plt.show()

# Macro F1 by window size
window_f1_df = combined_df[
    (combined_df["metric_group"] == "window_size") &
    (combined_df["metric_name"] == "macro_f1")
].copy()

window_f1_df["group_value"] = pd.to_numeric(window_f1_df["group_value"]).astype(int)
window_f1_df = window_f1_df.sort_values(["prompt_type", "group_value"])

plt.figure(figsize=(9, 6))
for prompt_type in window_f1_df["prompt_type"].unique():
    subset = window_f1_df[window_f1_df["prompt_type"] == prompt_type]
    plt.plot(subset["group_value"], subset["metric_value"], marker="o", label=prompt_type)

plt.xlabel("Window Size")
plt.ylabel("Macro F1")
plt.title("Macro F1 by Window Size Across Prompting Methods")
plt.xticks([1, 3, 5, 7, 9, 11])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT_DIR / "combined_macro_f1_by_window.png")
plt.show()

# Overall metrics grouped bar chart
overall_df = combined_df[combined_df["metric_group"] == "overall"].copy()

metric_order = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
overall_df["metric_name"] = pd.Categorical(
    overall_df["metric_name"],
    categories=metric_order,
    ordered=True
)
overall_df = overall_df.sort_values(["metric_name", "prompt_type"])

pivot_df = overall_df.pivot(
    index="metric_name",
    columns="prompt_type",
    values="metric_value"
).reindex(metric_order)

x = np.arange(len(pivot_df.index))
width = 0.25

plt.figure(figsize=(10, 6))
prompt_types = list(pivot_df.columns)

for i, prompt_type in enumerate(prompt_types):
    plt.bar(x + (i - 1) * width, pivot_df[prompt_type], width, label=prompt_type)

plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Overall Metrics Across Prompting Methods")
plt.xticks(x, pivot_df.index, rotation=15)
plt.ylim(0, 1)
plt.grid(True, axis="y")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT_DIR / "combined_overall_metrics_bar.png")
plt.show()