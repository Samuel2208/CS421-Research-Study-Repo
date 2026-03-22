from google import genai
from dotenv import load_dotenv
import pandas as pd
import random
from collections import defaultdict
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

FILENAME = f"least_to_most_results.csv"

VALID_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

load_dotenv()

client = genai.Client()


prompt = (
    """
    You are given a dialogue. Your task is to determine the emotion of the LAST utterance.

    Follow these steps:

    Step 1: Summarize each utterance briefly.
    Step 2: Describe how the emotions or tone evolve across the dialogue.
    Step 3: Analyze the final utterance in the context of previous ones.
    Step 4: Predict the emotion of the final utterance from this list:
    [neutral, joy, sadness, anger, fear, disgust, surprise]

    Return ONLY the final emotion.

    Dialogue:
    {dialogue_here}
    """
)



def load_data():
    df = pd.read_csv('../Dataset/MELD.csv')

    # Clean column names (sometimes MELD has weird characters)
    df.columns = df.columns.str.strip()

    # Sort properly
    df = df.sort_values(by=["Dialogue_ID", "Utterance_ID"])

    # Quick sanity check
    print("Unique emotions:", df["Emotion"].unique())
    print("Total dialogues:", df["Dialogue_ID"].nunique())

    return df

def build_dialogues(df):
    dialogues = {}

    for dialogue_id, group in df.groupby("Dialogue_ID"):
        group = group.sort_values("Utterance_ID")

        dialogues[dialogue_id] = [
            {
                "speaker": row["Speaker"],
                "utterance": row["Utterance"],
                "emotion": row["Emotion"]
            }
            for _, row in group.iterrows()
        ]

    return dialogues

def generate_windows(dialogues, window_sizes=[1,3,5,7,9,11]):
    samples = []

    for dialogue_id, utterances in dialogues.items():
        total_len = len(utterances)

        for k in window_sizes:
            if k <= total_len:
                window = utterances[:k]

                samples.append({
                    "dialogue_id": dialogue_id,
                    "window_size": k,
                    "context": window[:-1],   # everything before last
                    "target": window[-1],     # last utterance
                    "label": window[-1]["emotion"]
                })

    return samples

def format_dialogue(sample):
    lines = []

    # Add context
    for turn in sample["context"]:
        lines.append(f"{turn['speaker']}: {turn['utterance']}")

    # Add target (this is what we predict)
    target = sample["target"]
    lines.append(f"{target['speaker']}: {target['utterance']}")

    return "\n".join(lines)

def prepare_dataset(samples):
    dataset = []

    for sample in samples:
        dataset.append({
            "dialogue_id": sample["dialogue_id"],
            "window_size": sample["window_size"],
            "input_text": format_dialogue(sample),
            "label": sample["label"]
        })

    return dataset


def select_dialogues(dialogues, num_dialogues=28, seed=42):
    random.seed(seed)
    selected_ids = random.sample(list(dialogues.keys()), num_dialogues)
    return {k: dialogues[k] for k in selected_ids}


def query_gemini(client, prompt_text, model="gemini-2.5-pro"):
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text
        )
        return response.text.strip().lower()
    except Exception as e:
        print("Error:", e)
        return "error"
    
def run_experiment(dataset, client, max_samples=None):
    results = []

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        dialogue_text = sample["input_text"]

        full_prompt = prompt.format(dialogue_here=dialogue_text)

        raw_pred = query_gemini(client, full_prompt)
        prediction = clean_prediction(raw_pred)

        results.append({
            "dialogue_id": sample["dialogue_id"],
            "window_size": sample["window_size"],
            "prediction": prediction,
            "label": sample["label"],
            "prompt_type": "least_to_most",
            "model": "gemini"
        })

        if i % 10 == 0:
            print(f"Processed {i} samples")

    return results

def clean_prediction(pred):
    pred = pred.lower()

    for emotion in VALID_EMOTIONS:
        if emotion in pred:
            return emotion

    return "unknown"

def evaluate(results):
    correct = 0

    for r in results:
        if r["prediction"] == r["label"]:
            correct += 1

    accuracy = correct / len(results)
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def evaluate_by_window(results):
    grouped = defaultdict(list)

    for r in results:
        grouped[r["window_size"]].append(r)

    for window_size, items in grouped.items():
        acc = sum(1 for r in items if r["prediction"] == r["label"]) / len(items)
        print(f"Window {window_size}: {acc:.4f}")

def save_results(results, filename="results.csv"):
    df = pd.DataFrame(results)

    # If file exists, append (useful for multiple runs)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

    print(f"Saved {len(results)} rows to {filename}")

def load_results(filename="results.csv"):
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows from {filename}")
    return df

def evaluate_from_df(df):
    accuracy = (df["prediction"] == df["label"]).mean()
    print(f"Overall Accuracy: {accuracy:.4f}")
    return accuracy

def evaluate_by_window_df(df):
    grouped = df.groupby("window_size")

    for window_size, group in grouped:
        acc = (group["prediction"] == group["label"]).mean()
        print(f"Window {window_size}: {acc:.4f}")

if __name__ == "__main__":

    # Load + prepare
    df = load_data()
    dialogues = build_dialogues(df)

    # Limit to 28 dialogues
    dialogues = select_dialogues(dialogues, 28)

    samples = generate_windows(dialogues)
    dataset = prepare_dataset(samples)

    print(f"Total dataset size: {len(dataset)}")
    print(f"Total API calls to be made: {len(dataset)}")


    # Run experiment (start small!)
    # print("Running experiment...")
    # results = run_experiment(dataset, client, max_samples=20)
    # save_results(results, FILENAME)


    #Evaluation Move to a different file
    df = load_results(FILENAME)

    evaluate_from_df(df)
    evaluate_by_window_df(df)

