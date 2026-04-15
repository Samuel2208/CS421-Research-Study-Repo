from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Compute all metrics for a DataFrame.
def all_metrics(df):
    y_true = df["label"]
    y_pred = df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# Compute multiclass metrics by window size.
def compute_metrics_by_window(df):
    rows = []

    for window_size, group_df in df.groupby("window_size"):
        metrics = all_metrics(group_df)
        row = {"window_size": window_size}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("window_size")


# Compute one-vs-rest metrics by label.
def compute_metrics_by_label(df):
    rows = []

    labels = sorted(df["label"].unique())

    for label_name in labels:
        y_true = (df["label"] == label_name).astype(int)
        y_pred = (df["prediction"] == label_name).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0
        )

        rows.append({
            "label": label_name,
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        })

    return pd.DataFrame(rows).sort_values("label")


def main():
    input_file = RESULTS_DIR / "qwen_least_to_most_structured_results.csv"
    df = pd.read_csv(input_file)

    input_stem = input_file.stem

    # overall
    overall_metrics = all_metrics(df)
    overall_df = pd.DataFrame([overall_metrics])

    # by window size
    window_df = compute_metrics_by_window(df)

    # by label
    label_df = compute_metrics_by_label(df)

    # pretty terminal output
    print("\n--- Overall Metrics ---")
    print(overall_df.to_string(index=False))

    print("\n--- Metrics by Window Size ---")
    print(window_df.to_string(index=False))

    print("\n--- Metrics by Label ---")
    print(label_df.to_string(index=False))

    # save everything to one file in long format
    long_rows = []

    for metric_name, metric_value in overall_metrics.items():
        long_rows.append({
            "source_file": input_stem,
            "metric_group": "overall",
            "group_value": "all",
            "metric_name": metric_name,
            "metric_value": metric_value
        })

    for _, row in window_df.iterrows():
        for metric_name in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            long_rows.append({
                "source_file": input_stem,
                "metric_group": "window_size",
                "group_value": row["window_size"],
                "metric_name": metric_name,
                "metric_value": row[metric_name]
            })

    for _, row in label_df.iterrows():
        for metric_name in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            long_rows.append({
                "source_file": input_stem,
                "metric_group": "label",
                "group_value": row["label"],
                "metric_name": metric_name,
                "metric_value": row[metric_name]
            })

    results_df = pd.DataFrame(long_rows)

    output_file = RESULTS_DIR / f"{input_stem}_evaluation.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()