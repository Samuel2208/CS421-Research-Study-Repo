import os
import pandas as pd
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()


# Load the MELD CSV file into a pandas DataFrame.
def load_meld(path="/Users/dj/Desktop/CS421-Research-Study-Repo/Dataset/MELD.csv"): #need to fix this path later
    return pd.read_csv(path)


# Fix broken text encodings in MELD utterances, especially apostrophe contractions.
def clean_text(text):
    if pd.isna(text):
        return text

    text = str(text)

    replacements = {
        "Â¬Ãs": "'s",
        "Â¬Ãve": "'ve",
        "Â¬Ãll": "'ll",
        "Â¬Ãre": "'re",
        "Â¬Ãm": "'m",
        "Â¬Ãd": "'d",
        "Â¬Ã": "'",
        "¬ís": "'s",
        "¬íve": "'ve",
        "¬íll": "'ll",
        "¬íre": "'re",
        "¬ím": "'m",
        "¬íd": "'d",
        "¬í": "'",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


# Load MELD, keep only needed columns, sort dialogue order, and clean utterances.
def prepare_meld_dataframe(path="/Users/dj/Desktop/CS421-Research-Study-Repo/Dataset/MELD.csv"):
    df = load_meld(path)

    meld_df = df[["Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion"]].copy()
    meld_df = meld_df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    meld_df["Utterance"] = meld_df["Utterance"].apply(clean_text)

    return meld_df


# Return the target utterance with the requested amount of dialogue context.
# k=0 -> target only
# k=n -> previous n utterances + target
# k=None -> full dialogue history up to the target
def get_context_window(dataframe, dialogue_id, utterance_id, k=None):
    dialogue = dataframe[dataframe["Dialogue_ID"] == dialogue_id]
    dialogue = dialogue.sort_values("Utterance_ID").reset_index(drop=True)

    target_row = dialogue[dialogue["Utterance_ID"] == utterance_id]

    if target_row.empty:
        return None

    target_index = target_row.index[0]

    if k is None:
        context_df = dialogue.iloc[:target_index + 1]
    elif k == 0:
        context_df = dialogue.iloc[target_index:target_index + 1]
    else:
        start_index = max(0, target_index - k)
        context_df = dialogue.iloc[start_index:target_index + 1]

    return context_df


# Convert a context window DataFrame into prompt-ready dialogue text.
def format_context_for_prompt(context_df):
    if context_df is None or context_df.empty:
        return ""

    lines = []

    for _, row in context_df.iterrows():
        speaker = row["Speaker"]
        utterance = row["Utterance"]
        lines.append(f"{speaker}: {utterance}")

    return "\n".join(lines)


# Build the zero-shot prompt for Gemini using the dialogue context and target utterance.
def build_zero_shot_prompt(context_text, target_utterance):
    prompt = f"""You are an emotion classification assistant.

Your task is to classify the emotion of the target utterance in the dialogue below.

Choose exactly one label from this list:
anger, disgust, fear, joy, neutral, sadness, surprise

Dialogue:
{context_text}

Target utterance:
{target_utterance}

Answer with only one emotion label.
"""
    return prompt


# Send the prompt to Gemini and return the model's response text.
def get_gemini_prediction(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in your environment.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()

# Use this in case we get lengthy and results that aren't one word. Normalize Gemini's raw response so it matches one of the valid MELD labels.
def normalize_prediction(prediction):
    valid_labels = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}

    if not prediction:
        return "INVALID"

    cleaned = prediction.strip().lower()

    if cleaned in valid_labels:
        return cleaned

    for label in valid_labels:
        if label in cleaned:
            return label

    return "INVALID"

# Run the full zero-shot pipeline for one utterance.
def classify_meld_utterance(dataframe, dialogue_id, utterance_id, k=None):
    context_df = get_context_window(dataframe, dialogue_id, utterance_id, k)

    if context_df is None or context_df.empty:
        return None

    formatted_context = format_context_for_prompt(context_df)
    target_utterance = context_df.iloc[-1]["Utterance"]
    true_label = context_df.iloc[-1]["Emotion"]

    prompt = build_zero_shot_prompt(formatted_context, target_utterance)
    prediction = get_gemini_prediction(prompt)

    return {
        "Dialogue_ID": dialogue_id,
        "Utterance_ID": utterance_id,
        "k": "full_context" if k is None else k,
        "target_utterance": target_utterance,
        "true_label": true_label,
        "prediction": prediction
    }

# Run one truncation condition across the dataset and return a results DataFrame.
# def run_experiment_for_k(dataframe, k, max_rows=None):
#     results = []

#     if max_rows is not None:
#         rows_to_process = dataframe.head(max_rows)
#     else:
#         rows_to_process = dataframe

#     for _, row in rows_to_process.iterrows():
#         result = classify_meld_utterance(
#             dataframe,
#             dialogue_id=row["Dialogue_ID"],
#             utterance_id=row["Utterance_ID"],
#             k=k
#         )

#         if result:
#             results.append(result)

#     return pd.DataFrame(results)


def run_experiment_for_k_resumable(dataframe, k, output_file, max_rows=None, sleep_seconds=0):
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        completed = len(existing_df)
        print(f"Resuming {output_file} from row {completed}...")
    else:
        completed = 0
        print(f"Starting new file: {output_file}")

    if max_rows is not None:
        rows_to_process = dataframe.iloc[:max_rows]
    else:
        rows_to_process = dataframe

    total_rows = len(rows_to_process)

    for idx in range(completed, total_rows):
        row = rows_to_process.iloc[idx]

        try:
            result = classify_meld_utterance(
                dataframe,
                dialogue_id=row["Dialogue_ID"],
                utterance_id=row["Utterance_ID"],
                k=k
            )

            if result:
                result_df = pd.DataFrame([result])

                if os.path.exists(output_file):
                    result_df.to_csv(output_file, mode="a", header=False, index=False)
                else:
                    result_df.to_csv(output_file, index=False)

                print(f"Saved row {idx + 1}/{total_rows} for k={k}")

            time.sleep(sleep_seconds)

        except Exception as e:
            print(f"Stopped at row {idx} for k={k} due to error: {e}")
            break


# def main():
#     meld_df = prepare_meld_dataframe()

#     # Truncation settings for the study
#     truncation_settings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None]

#     all_results = []

#     for k in truncation_settings:
#         if k is None:
#             print("\nRunning full context condition...")
#             file_name = "results/results_full_context.csv"
#         else:
#             print(f"\nRunning k={k} condition...")
#             file_name = f"results/results_k{k}.csv"

#         results_df = run_experiment_for_k(meld_df, k=k, max_rows=5)

#         print(results_df.head())

#         results_df.to_csv(file_name, index=False)
#         print(f"Saved: {file_name}")

#         all_results.append(results_df)

#     combined_results_df = pd.concat(all_results, ignore_index=True)
#     combined_results_df.to_csv("results/results_all_conditions.csv", index=False)
#     print("\nSaved: results_all_conditions.csv")

def main():
    meld_df = prepare_meld_dataframe()

    # Choose one truncation setting at a time.
    # Examples: 0, 1, 2, 12, or None for full context
    k = 0

    if k is None:
        output_file = "results/results_full_context.csv"
    else:
        output_file = f"results/results_k{k}.csv"

    run_experiment_for_k_resumable(
        dataframe=meld_df,
        k=0,
        output_file=output_file,
        max_rows=None,
        sleep_seconds=0
    )


if __name__ == "__main__":
    main()
