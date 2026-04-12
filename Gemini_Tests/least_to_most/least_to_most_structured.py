import pprint

from google import genai
from dotenv import load_dotenv
import pandas as pd
import random
from collections import defaultdict
import os
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

RESULTS_FILE = f"./results/least_to_most_results.csv"
DATASET_FILE = "../../Dataset/MELD_56_dialogues.csv"

VALID_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

load_dotenv()

client = genai.Client()


prompt = """
You are given structured dialogue data.

Follow these steps internally:
1. Summarize each utterance briefly.
2. Describe how emotions evolve.
3. Analyze the final utterance in context.
4. Choose the final emotion.

IMPORTANT:
- Return ONLY valid JSON
- Use DOUBLE quotes
- No markdown

Input:
{structured_input}

Output format:
{{
    "reasoning": "...",
    "emotion": "..."
}}
"""



def load_data():
    df = pd.read_csv(DATASET_FILE)

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

def generate_windows(dialogues, window_sizes=[1,3,5,7,9,11], max_target_index=11):
    samples = []

    for dialogue_id, utterances in dialogues.items():
        total_len = len(utterances)

        # Determine target index
        target_idx = min(max_target_index, total_len - 1)

        for k in window_sizes:
            if k <= (target_idx + 1):  # ensure enough context

                # Get window ending at target_idx
                start_idx = target_idx - k + 1
                window = utterances[start_idx:target_idx + 1]

                samples.append({
                    "dialogue_id": dialogue_id,
                    "window_size": k,
                    "context": window[:-1],   # previous context
                    "target": window[-1],     # ALWAYS the same target
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
    emotion_count = defaultdict(int)

    for sample in samples:
        # print("Example dialogue:", "dialogue_id =", sample["dialogue_id"], "window_size =", sample["window_size"])
        # print(format_dialogue(sample))
        # print()
        dataset.append({
            "dialogue_id": sample["dialogue_id"],
            "window_size": sample["window_size"],
            "input_text": format_dialogue(sample),
            "label": sample["label"].lower(),

            "context": sample["context"],
            "target": sample["target"]
        })
        emotion_count[sample["label"]] += 1

    return dataset, set(emotion_count.keys()), emotion_count


def select_dialogues(dialogues, num_dialogues=28, seed=42):
    random.seed(seed)
    selected_ids = random.sample(list(dialogues.keys()), num_dialogues)
    return {k: dialogues[k] for k in selected_ids}


def query_gemini(client, prompt_text, model="gemini-3-flash-preview"):
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text,
            config={
                "response_mime_type": "application/json"
            }
        )
        return response.text.strip()
    except Exception as e:
        print("Error:", e)
        return "error"

def parse_llm_output(raw_pred):
    if raw_pred == "error":
        return "unknown", "api_error"

    try:
        data = json.loads(raw_pred)
        emotion = data.get("emotion", "unknown").lower()
        reasoning = data.get("reasoning", "")
        return emotion, reasoning
    except Exception as e:
        print("JSON parse error:", e)
        return "unknown", raw_pred
    
def run_experiment(dataset, client, max_samples=None):
    results = []

    count = 0

    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # dialogue_text = sample["input_text"]

        # full_prompt = prompt.format(dialogue_here=dialogue_text)

        structured_input = format_structured_dialogue(sample)
        full_prompt = prompt.format(
            structured_input=json.dumps(structured_input, indent=2)
        )

        print("Prompt example:")
        pprint.pprint(full_prompt)



        raw_pred = query_gemini(client, full_prompt)
        # print(f"Raw prediction: {raw_pred}")
        # print()
        prediction, reasoning = parse_llm_output(raw_pred)
        # print()
        # print(f"Parsed prediction: {prediction}, Reasoning: {reasoning}")

        results.append({
            "dialogue_id": sample["dialogue_id"],
            "window_size": sample["window_size"],
            "prediction": prediction,
            "label": sample["label"].lower(),
            "prompt_type": "least_to_most_structured",
            "model": "gemini",
            "reasoning": reasoning,
            "accuracy": prediction == sample["label"],
            "prompt_word_count": len(full_prompt.split()),
            "prompt_character_count" : len(full_prompt)
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

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved {len(results)} rows to {filename}")

def load_results(filename):
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


def format_structured_dialogue(sample):
    return {
        "context": [
            {
                "speaker": turn["speaker"],
                "utterance": turn["utterance"]
            }
            for turn in sample["context"]
        ],
        "target_utterance": {
            "speaker": sample["target"]["speaker"],
            "utterance": sample["target"]["utterance"]
        },
        "task": "Classify the emotion of the target utterance",
        "emotion_options": VALID_EMOTIONS
    }

#Run experiment paralle making it run faster test function
def run_experiment_parallel(dataset, client, max_workers=5):
    results = []

    def process_sample(sample):
        structured_input = format_structured_dialogue(sample)

        full_prompt = prompt.format(
            structured_input=json.dumps(structured_input, indent=2)
        )

        raw_pred = query_gemini(client, full_prompt)
        prediction, reasoning = parse_llm_output(raw_pred)

        return {
            "dialogue_id": sample["dialogue_id"],
            "window_size": sample["window_size"],
            "prediction": prediction,
            "label": sample["label"].lower(),
            "prompt_type": "least_to_most_structured",
            "model": "gemini",
            "reasoning": reasoning,
            "accuracy": prediction == sample["label"],
            "prompt_word_count": len(full_prompt.split()),
            "prompt_character_count": len(full_prompt),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in dataset]

        for i, future in enumerate(as_completed(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print("Error in thread:", e)

            if i % 10 == 0:
                print(f"Processed {i} samples")

    return results



######################################### MAIN #########################################
def main():
    # Load + prepare
    df = load_data()
    dialogues = build_dialogues(df)

    # Limit to 28 dialogues
    # dialogues = select_dialogues(dialogues, 28)

    samples = generate_windows(dialogues)
    dataset, emotions, emotion_count = prepare_dataset(samples)
    print(dataset[0].keys())
    # return 


    print(f"Total dataset size: {len(dataset)}")
    print(f"Total API calls to be made: {len(dataset)}")
    pprint.pprint(emotion_count)

    # Run experiment (start small!)
    print("Running experiment...")
    results = run_experiment(dataset, client, max_samples=1)
    results_parallel = run_experiment_parallel(dataset[:10], client, max_workers=5)
    # save_results(results, RESULTS_FILE)


    # #Evaluation Move to a different file
    # df = load_results(RESULTS_FILE)

    # evaluate_from_df(df)
    # evaluate_by_window_df(df)



if __name__ == "__main__":
    main()

    
