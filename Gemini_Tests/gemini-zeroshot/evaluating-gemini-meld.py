from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Load one standardized results CSV and calculate evaluation metrics.
def evaluate_results(file_path):
    df = pd.read_csv(file_path)

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
        "file": Path(file_path).name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


def main():
    # can exapand if we all want to use the same one just add the file path here
    result_files = [
        RESULTS_DIR / "gemini_zero_shot_results.csv"
    ]

    all_metrics = []

    for file_path in result_files:
        metrics = evaluate_results(file_path)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    print("\n--- Evaluation Results ---")
    print(metrics_df.to_string(index=False))

    output_path = RESULTS_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()