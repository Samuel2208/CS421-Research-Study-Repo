from pathlib import Path
import os
import pandas as pd
import re
from vllm import LLM, SamplingParams

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
DATASET_DIR = REPO_DIR / "Dataset"
SUBSET_PATH = DATASET_DIR / "MELD_filtered_dialogues.csv"


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


# Build the zero-shot prompt messages using the dialogue context and target utterance.
def build_zero_shot_prompt_messages(context_text, target_utterance):
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
    return [{"role": "user", "content": prompt}]


# Split the model's response into reasoning text and final label.
def parse_vllm_response(response_text):
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

    # Fallback in case the model doesn't follow the exact format
    if prediction == "INVALID":
        prediction = normalize_prediction(response_text)

    return reasoning, prediction


# Use this in case we get lengthy results that aren't one word. 
# Normalize raw response so it matches one of the valid MELD labels.
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


# Run zero-shot experiments over selected dialogues and window sizes using vLLM batched inference.
def run_vllm_zero_shot(dataframe, output_file, window_sizes, max_dialogues=None):
    print("Loading model into VRAM...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ", 
        quantization="awq",
        max_model_len=4096
    )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    dialogue_ids = sorted(dataframe["Dialogue_ID"].unique())
    if max_dialogues is not None:
        dialogue_ids = dialogue_ids[:max_dialogues]

    prompts = []
    metadata = []

    print("Building prompts...")
    for dialogue_id in dialogue_ids:
        for window_size in window_sizes:
            window_df = get_last_window(dataframe, dialogue_id, window_size)
            
            if window_df is None or window_df.empty:
                continue

            formatted_context = format_context_for_prompt(window_df)
            target_utterance = window_df.iloc[-1]["Utterance"]
            true_label = window_df.iloc[-1]["Emotion"]

            messages = build_zero_shot_prompt_messages(formatted_context, target_utterance)
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            prompts.append(formatted_prompt)
            metadata.append({
                "dialogue_id": dialogue_id,
                "window_size": window_size,
                "label": true_label,
                "prompt_character_count": len(formatted_prompt),
                "prompt_word_count": len(formatted_prompt.split())
            })

    print(f"Running batched inference on {len(prompts)} prompts simultaneously...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    print("Parsing results...")
    
    for i, output in enumerate(outputs):
        raw_output = output.outputs[0].text.strip()
        meta = metadata[i]

        reasoning, prediction = parse_vllm_response(raw_output)
        accuracy = prediction == meta["label"].lower()

        results.append({
            "dialogue_id": meta["dialogue_id"],
            "window_size": meta["window_size"],
            "prediction": prediction,
            "label": meta["label"],
            "prompt_type": "zero_shot",
            "model": "qwen2.5-7b-instruct-awq",
            "reasoning": reasoning,
            "accuracy": accuracy,
            "prompt_word_count": meta["prompt_word_count"],
            "prompt_character_count": meta["prompt_character_count"]
        })

    results_df = pd.DataFrame(results)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_file, index=False)
    print(f"Finished! Results saved to {output_file}")


def main():
    meld_df = prepare_meld_dataframe()

    output_file = "results/qwen_vllm_zero_shot_results.csv"
    window_sizes = [1, 3, 5, 7, 9, 11]

    run_vllm_zero_shot(
        dataframe=meld_df,
        output_file=output_file,
        window_sizes=window_sizes,
        max_dialogues=None
    )


if __name__ == "__main__":
    main()