from pathlib import Path
import os
import json
import pandas as pd
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
DATASET_DIR = REPO_DIR / "Dataset"
SUBSET_PATH = DATASET_DIR / "MELD_56_dialogues.csv"

VALID_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


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


# Get the last `window_size` utterances from a dialogue.
# The target for prediction will always be the last utterance in this window.
def get_last_window(dataframe, dialogue_id, window_size):
    dialogue = dataframe[dataframe["Dialogue_ID"] == dialogue_id]
    dialogue = dialogue.sort_values("Utterance_ID").reset_index(drop=True)

    if dialogue.empty:
        return None

    window_df = dialogue.tail(window_size).reset_index(drop=True)

    return window_df


# Convert the window into a structured JSON-like object for prompting.
def format_structured_dialogue(window_df):
    if window_df is None or window_df.empty:
        return {}

    context_turns = []

    for _, row in window_df.iloc[:-1].iterrows():
        context_turns.append({
            "speaker": row["Speaker"],
            "utterance": row["Utterance"]
        })

    target_row = window_df.iloc[-1]

    structured_input = {
        "task": "Classify the emotion of the target utterance",
        "emotion_options": VALID_EMOTIONS,
        "context": context_turns,
        "target_utterance": {
            "speaker": target_row["Speaker"],
            "utterance": target_row["Utterance"]
        }
    }

    return structured_input


# Build the structured zero-shot prompt for Gemini.
def build_structured_zero_shot_prompt(structured_input):
    prompt = f"""
You are given structured dialogue data.

Classify the emotion of the target utterance.

IMPORTANT:
- Use the dialogue context only if it helps interpret the target utterance
- Focus on the emotional meaning of the target utterance
- Choose exactly one emotion from the provided emotion options
- Return ONLY valid JSON
- Use DOUBLE quotes
- Do not return markdown
- Do not return any extra text outside the JSON object

Input:
{structured_input}

Output format:
{{
    "reasoning": "...",
    "emotion": "..."
}}
"""
    return prompt.strip()


# Send the prompt to Gemini and return the model's response text.
def get_gemini_prediction(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in your environment.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    return response.text.strip()


# Split Gemini's response into reasoning text and final label.
def parse_model_response(response_text):
    reasoning = ""
    prediction = "INVALID"

    if not response_text:
        return reasoning, prediction

    try:
        data = json.loads(response_text)
        reasoning = str(data.get("reasoning", "")).strip()
        prediction = str(data.get("emotion", "INVALID")).strip().lower()
        prediction = normalize_prediction(prediction)
        return reasoning, prediction

    except Exception:
        lines = response_text.strip().splitlines()

        for line in lines:
            lower_line = line.lower().strip()

            if lower_line.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

            elif lower_line.startswith("label:") or lower_line.startswith("emotion:"):
                prediction = line.split(":", 1)[1].strip().lower()

        if prediction == "INVALID":
            prediction = normalize_prediction(response_text)

        return reasoning, prediction


# Normalize Gemini's raw response so it matches one of the valid MELD labels.
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


# Run the structured zero-shot pipeline for one dialogue window.
# The target is always the last utterance in the window.
def classify_dialogue_window(dataframe, dialogue_id, window_size, prompt_type="structured_zero_shot", model_name="gemini"):
    window_df = get_last_window(dataframe, dialogue_id, window_size)

    if window_df is None or window_df.empty:
        return None

    true_label = window_df.iloc[-1]["Emotion"]
    structured_input = format_structured_dialogue(window_df)

    prompt = build_structured_zero_shot_prompt(
        json.dumps(structured_input, indent=2)
    )

    prompt_length = len(prompt)
    prompt_word_count = len(prompt.split())

    raw_response = get_gemini_prediction(prompt)

    reasoning, prediction = parse_model_response(raw_response)
    accuracy = prediction == true_label

    return {
        "dialogue_id": dialogue_id,
        "window_size": window_size,
        "prediction": prediction,
        "label": true_label,
        "prompt_type": prompt_type,
        "model": model_name,
        "reasoning": reasoning,
        "accuracy": accuracy,
        "prompt_word_count": prompt_word_count,
        "prompt_character_count": prompt_length
    }


# Run structured zero-shot experiments over selected dialogues and window sizes.
# Saves progress after each result so the run can resume later.
def run_zero_shot_resumable(dataframe, output_file, window_sizes, max_dialogues=None, sleep_seconds=0):
    dialogue_ids = sorted(dataframe["Dialogue_ID"].unique())

    if max_dialogues is not None:
        dialogue_ids = dialogue_ids[:max_dialogues]

    tasks = []
    for dialogue_id in dialogue_ids:
        for window_size in window_sizes:
            tasks.append((dialogue_id, window_size))

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
                prompt_type="structured_zero_shot",
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

    output_file = "structured-results/gemini_structured_zero_shot_results.csv"
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