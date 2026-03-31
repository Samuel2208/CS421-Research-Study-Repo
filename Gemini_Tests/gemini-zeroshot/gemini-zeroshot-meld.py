from pathlib import Path
import os
import pandas as pd
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
DATASET_DIR = REPO_DIR / "Dataset"
SUBSET_PATH = DATASET_DIR / "MELD_28_dialogues.csv"


# Load the reduced MELD subset CSV file into a pandas DataFrame.
def load_meld_subset(path=SUBSET_PATH):
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


# Load the reduced MELD subset, keep only needed columns, sort dialogue order, and clean utterances.
def prepare_meld_dataframe(path=SUBSET_PATH):
    df = load_meld_subset(path)

    meld_df = df[["Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion"]].copy()
    meld_df = meld_df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    meld_df["Utterance"] = meld_df["Utterance"].apply(clean_text)

    return meld_df

# Return the target utterance with the requested amount of dialogue context.
# k=0 -> target only
# k=n -> previous n utterances + target
# k=None -> full dialogue history up to the target
# Get the last `window_size` utterances from a dialogue.
# The target for prediction will always be the last utterance in this window.
def get_last_window(dataframe, dialogue_id, window_size):
    dialogue = dataframe[dataframe["Dialogue_ID"] == dialogue_id]
    dialogue = dialogue.sort_values("Utterance_ID").reset_index(drop=True)

    if dialogue.empty:
        return None

    # Take the last `window_size` utterances from the dialogue
    window_df = dialogue.tail(window_size).reset_index(drop=True)

    return window_df


# Convert a context window DataFrame into prompt-ready dialogue text.
def format_context_for_prompt(context_df):
    if context_df is None or context_df.empty:
        return ""

    lines = []

    # Format each utterance as "Speaker: Utterance" and join them with newlines
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

Explain your reasoning in one or two sentences behind your decision and then choose one label from the list above.

Return your answer in this format:
Reasoning: <your reasoning>
Label: <one emotion label>
"""
    return prompt


# Send the prompt to Gemini and return the model's response text.
def get_gemini_prediction(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in your environment.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text.strip()

# Split Gemini's response into reasoning text and final label.
def parse_gemini_response(response_text):
    reasoning = ""
    prediction = "INVALID"

    if not response_text:
        return reasoning, prediction

    lines = response_text.strip().splitlines()

    for line in lines:
        lower_line = line.lower().strip()

        if lower_line.startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

        elif lower_line.startswith("label:"):
            prediction = line.split(":", 1)[1].strip().lower()

    # Fallback in case Gemini doesn't follow the exact format
    if prediction == "INVALID":
        prediction = normalize_prediction(response_text)

    return reasoning, prediction

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

# Run the zero-shot pipeline for one dialogue window.
# The target is always the last utterance in the window.
def classify_dialogue_window(dataframe, dialogue_id, window_size, prompt_type="zero_shot", model_name="gemini"):
    window_df = get_last_window(dataframe, dialogue_id, window_size)

    if window_df is None or window_df.empty:
        return None

    formatted_context = format_context_for_prompt(window_df)
    target_utterance = window_df.iloc[-1]["Utterance"]
    true_label = window_df.iloc[-1]["Emotion"]

    prompt = build_zero_shot_prompt(formatted_context, target_utterance)
    raw_response = get_gemini_prediction(prompt)

    reasoning, prediction = parse_gemini_response(raw_response)
    accuracy = prediction == true_label

    return {
        "dialogue_id": dialogue_id,
        "window_size": window_size,
        "prediction": prediction,
        "label": true_label,
        "prompt_type": prompt_type,
        "model": model_name,
        "reasoning": reasoning,
        "accuracy": accuracy
    }

# Run zero-shot experiments over selected dialogues and window sizes.
# Saves progress after each result so the run can resume later.
def run_zero_shot_resumable(dataframe, output_file, window_sizes, max_dialogues=None, sleep_seconds=0):
    dialogue_ids = sorted(dataframe["Dialogue_ID"].unique())

    if max_dialogues is not None:
        dialogue_ids = dialogue_ids[:max_dialogues]

    # Build the full list of tasks: one task per (dialogue_id, window_size)
    tasks = []
    for dialogue_id in dialogue_ids:
        for window_size in window_sizes:
            tasks.append((dialogue_id, window_size))

    # Check how many tasks are already saved
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        completed = len(existing_df)
        print(f"Resuming {output_file} from task {completed}...")
    else:
        completed = 0
        print(f"Starting new file: {output_file}")

    total_tasks = len(tasks)

    for idx in range(completed, total_tasks):
        dialogue_id, window_size = tasks[idx]

        try:
            result = classify_dialogue_window(
                dataframe=dataframe,
                dialogue_id=dialogue_id,
                window_size=window_size,
                prompt_type="zero_shot",
                model_name="gemini"
            )

            if result:
                result_df = pd.DataFrame([result])

                if os.path.exists(output_file):
                    result_df.to_csv(output_file, mode="a", header=False, index=False)
                else:
                    result_df.to_csv(output_file, index=False)

                print(f"Saved task {idx + 1}/{total_tasks}: dialogue_id={dialogue_id}, window_size={window_size}")

            time.sleep(sleep_seconds)

        except Exception as e:
            print(f"Stopped at task {idx} (dialogue_id={dialogue_id}, window_size={window_size}) due to error: {e}")
            break


def main():
    meld_df = prepare_meld_dataframe()

    output_file = "results/gemini_zero_shot_results.csv"
    window_sizes = [1, 3, 5, 7, 9, 11]

    run_zero_shot_resumable(
        dataframe=meld_df,
        output_file=output_file,
        window_sizes=window_sizes,
        max_dialogues=None,
        sleep_seconds=0
    )


if __name__ == "__main__":
    main()
