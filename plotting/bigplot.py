from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

#exchange the paths as needed for the gemma and qwen tests
# NOVEL_APROACH = "structured"
# NOVEL_APROACH = "lemmatized"
NOVEL_APROACH = "structured_lemmatized"

# DATASET = "_dailydialog"
DATASET = "" #MELD



# LLM_NAME = "Gemma"
# LLM_DIRECTORY = "gemma_tests"

LLM_NAME = "Qwen2"
LLM_DIRECTORY = "qwen2_tests"

LLM_DISPLAY = LLM_NAME if LLM_NAME == "Gemma" else LLM_NAME.lower()[:-1]


COLORS = ['C0', 'C1', 'C2']
files = [
    {
        "name": f"{LLM_NAME} Zero-Shot",
        "method": 0,
        "aproach": "Baseline",
        "path": ROOT_DIR / LLM_DIRECTORY / "zero_shot" / "results" / f"{LLM_DISPLAY}_zero_shot{DATASET}_results_evaluation.csv"
    },
    {
        "name": f"{LLM_NAME} {NOVEL_APROACH} Zero-Shot",
        "method": 0,
        "aproach": NOVEL_APROACH,
        "path": ROOT_DIR / LLM_DIRECTORY / "zero_shot" / "results" / f"{LLM_DISPLAY}_zero_shot_{NOVEL_APROACH.lower()}{DATASET}_results_evaluation.csv"
    },
    {
        "name": f"{LLM_NAME} 2-Shot",
        "method": 1,
        "aproach": "Baseline",
        "path": ROOT_DIR / LLM_DIRECTORY / "2_shot" / "results" / f"{LLM_DISPLAY}_2_shot{DATASET}_results_evaluation.csv"
    },
    {
        "name": f"{LLM_NAME} {NOVEL_APROACH} 2-Shot",
        "method": 1,
        "aproach": NOVEL_APROACH,
        "path": ROOT_DIR / LLM_DIRECTORY / "2_shot" / "results" / f"{LLM_DISPLAY}_2_shot_{NOVEL_APROACH.lower()}{DATASET}_results_evaluation.csv"
    },
    {
        "name": f"{LLM_NAME} Least-to-Most",
        "method": 2,
        "aproach": "Baseline",
        "path": ROOT_DIR / LLM_DIRECTORY / "least_to_most" / "results" / f"{LLM_DISPLAY}_least_to_most{DATASET}_results_evaluation.csv"
    },
    {
        "name": f"{LLM_NAME} {NOVEL_APROACH} Least-to-Most",
        "method": 2,
        "aproach": NOVEL_APROACH,
        "path": ROOT_DIR / LLM_DIRECTORY / "least_to_most" / "results" / f"{LLM_DISPLAY}_least_to_most_{NOVEL_APROACH.lower()}{DATASET}_results_evaluation.csv"
    }
]

# files = [
#     {
#         "name": "Qwen Zero-Shot",
#         "path": ROOT_DIR / "qwen_tests" / "zero_shot" / "results" / "qwen_zero_shot_results_evaluation.csv"
#     },
#     {
#         "name": "Qwen Structured Zero-Shot",
#         "path": ROOT_DIR / "qwen_tests" / "zero_shot" / "results" / "qwen_zero_shot_structured_results_evaluation.csv"
#     },
#     {
#         "name": "Qwen 2-Shot",
#         "path": ROOT_DIR / "qwen_tests" / "2_shot" / "results" / "qwen_2_shot_results_evaluation.csv"
#     },
#     {
#         "name": "Qwen Structured 2-Shot",
#         "path": ROOT_DIR / "qwen_tests" / "2_shot" / "results" / "qwen_2_shot_structured_results_evaluation.csv"
#     },
#     {
#         "name": "Qwen Least-to-Most",
#         "path": ROOT_DIR / "qwen_tests" / "least_to_most" / "results" / "qwen_least_to_most_results_evaluation.csv"
#     },
#     {
#         "name": "Qwen Structured Least-to-Most",
#         "path": ROOT_DIR / "qwen_tests" / "least_to_most" / "results" / "qwen_least_to_most_structured_results_evaluation.csv"
#     }
# ]


# files = [
#     {
#         "name": "Gemini Zero-Shot",
#         "path": ROOT_DIR / "Gemini_Tests" / "gemini-zeroshot" / "baseline-results" / "gemini_zero_shot_results_evaluation.csv"
#     },
#     {
#         "name": "Gemini Structured Zero-Shot",
#         "path": ROOT_DIR / "Gemini_Tests" / "gemini-zeroshot" / "structured-results" / "gemini_structured_zero_shot_results_evaluation.csv"
#     },
#     {
#         "name": "Gemini 2-Shot",
#         "path": ROOT_DIR / "Gemini_Tests" / "gemini_2shot" / "results" / "gemini_meld_emotion_results_evaluation.csv"
#     },
#     {
#         "name": "Gemini Structured 2-Shot",
#         "path": ROOT_DIR / "Gemini_Tests" / "gemini_2shot" / "results" / "gemini_meld_emotion_results_structured_evaluation.csv"
#     },
#     {
#         "name": "Gemini Least-to-Most",
#         "path": ROOT_DIR / "Gemini_Tests" / "least_to_most" / "results" / "baseline" / "least_to_most_results_evaluation.csv"
#     },
#     {
#         "name": "Gemini Structured Least-to-Most",
#         "path": ROOT_DIR / "Gemini_Tests" / "least_to_most" / "results" / "structured_prompt" / "least_to_most_results_evaluation.csv"
#     }
# ]


SAVE_DIR = BASE_DIR / "MELD-plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_metric_by_window(file_path, metric_name):
    df = pd.read_csv(file_path)

    window_df = df[df["metric_group"] == "window_size"].copy()
    metric_df = window_df[window_df["metric_name"] == metric_name].copy()

    metric_df["group_value"] = metric_df["group_value"].astype(float).astype(int)
    metric_df = metric_df.sort_values("group_value")

    return metric_df[["group_value", "metric_value"]]


def add_min_max_labels_by_window(method_dfs, decimals=3):
    grouped = {}

    for method_df in method_dfs:
        for x, y in zip(method_df["group_value"], method_df["metric_value"]):
            if x not in grouped:
                grouped[x] = []
            grouped[x].append(y)

    for x in sorted(grouped.keys()):
        yvals = grouped[x]
        ymin = min(yvals)
        ymax = max(yvals)

        plt.text(
            x,
            ymax + 0.008,
            f"{ymax:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

        plt.text(
            x,
            ymin - 0.008,
            f"{ymin:.{decimals}f}",
            ha="center",
            va="top",
            fontsize=9
        )


for file_info in files:
    print(file_info["name"], "exists:", file_info["path"].exists())


# gemini accuracy by window size across prompting methods
plt.figure(figsize=(11, 6))

method_dfs = []

for file_info in files:
    metric_df = load_metric_by_window(file_info["path"], "accuracy")
    method_dfs.append(metric_df)

    if file_info["aproach"] == "Baseline":
        plt.plot(
            metric_df["group_value"],
            metric_df["metric_value"],
            marker="o",
            label=file_info["name"],
            linestyle="--",
            color = COLORS[file_info["method"]]
        )
    else:
        plt.plot(
            metric_df["group_value"],
            metric_df["metric_value"],
            marker="o",
            label=file_info["name"],
            color = COLORS[file_info["method"]]

        )

add_min_max_labels_by_window(method_dfs)

plt.xlabel("Window Size")
plt.ylabel("Accuracy")
plt.title(f"{LLM_NAME} Accuracy by Window Size Across Prompting Methods - {NOVEL_APROACH} vs Baseline - {DATASET if DATASET else 'MELD'}")
plt.xticks([1, 3, 5, 7, 9, 11])
# plt.ylim(0.30, 0.6)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / LLM_NAME.lower() / f"{LLM_NAME.lower()}_{NOVEL_APROACH.lower()}_accuracy_by_window_all_methods.png")
# plt.show()


# macro f1 by window size across prompting methods
plt.figure(figsize=(11, 6))

method_dfs = []

for file_info in files:
    metric_df = load_metric_by_window(file_info["path"], "macro_f1")
    method_dfs.append(metric_df)

    if file_info["aproach"] == "Baseline":
        plt.plot(
            metric_df["group_value"],
            metric_df["metric_value"],
            marker="o",
            label=file_info["name"],
            linestyle="--",
            color = COLORS[file_info["method"]]
        )
    else:
        plt.plot(
            metric_df["group_value"],
            metric_df["metric_value"],
            marker="o",
            label=file_info["name"],
            color = COLORS[file_info["method"]]
        )

add_min_max_labels_by_window(method_dfs)

plt.xlabel("Window Size")
plt.ylabel("Macro F1")
plt.title(f"{LLM_NAME} Macro F1 by Window Size Across Prompting Methods - {NOVEL_APROACH} vs Baseline - {DATASET if DATASET else 'MELD'}")
plt.xticks([1, 3, 5, 7, 9, 11])
# plt.ylim(0.10, 0.55)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / LLM_NAME.lower() / f"{LLM_NAME.lower()}_{NOVEL_APROACH.lower()}_macro_f1_by_window_all_methods.png")
plt.show()