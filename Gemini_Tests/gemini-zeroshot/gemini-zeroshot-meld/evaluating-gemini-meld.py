import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Load one results CSV and calculate evaluation metrics.
def evaluate_results(file_path):
    df = pd.read_csv(file_path)

    y_true = df["true_label"]
    y_pred = df["prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "file": file_path,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


def main():
    result_files = [
        "results/results_k0.csv",
        "results/results_k1.csv",
        "results/results_k2.csv",
        "results/results_k3.csv",
        "results/results_k4.csv",
        "results/results_k5.csv",
        "results/results_k6.csv",
        "results/results_k7.csv",
        "results/results_k8.csv",
        "results/results_k9.csv",
        "results/results_k10.csv",
        "results/results_k11.csv",
        "results/results_k12.csv",
        "results/results_full_context.csv"
    ]

    all_metrics = []

    for file_name in result_files:
        metrics = evaluate_results(file_name)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    print("\n--- Evaluation Results ---")
    print(metrics_df)

    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    print("\nSaved: evaluation_metrics.csv")


if __name__ == "__main__":
    main()