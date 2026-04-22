from pathlib import Path
import os
import json
import pandas as pd
import re
from vllm import LLM, SamplingParams
import sys

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
sys.path.append(str(REPO_DIR))
from utils import lexicon_analysis

DATASET_DIR = REPO_DIR / "Dataset"
SUBSET_PATH = DATASET_DIR / "DailyDialog_filtered_dialogues.csv"

VALID_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def load_subset(path=SUBSET_PATH):
    return pd.read_csv(path)

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

def prepare_dataframe(path=SUBSET_PATH):
    df = load_subset(path)

    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    df["Speaker"] = df["Utterance_ID"].apply(lambda x: "A" if x % 2 == 0 else "B")
    df["Utterance"] = df["Utterance"].apply(clean_text)

    return df

def parse_vllm_response(response_text):
    match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
    json_str = match.group(0) if match else response_text.strip()

    try:
        data = json.loads(json_str)
        prediction = str(data.get("emotion", "INVALID")).strip().lower()
        reasoning = str(data.get("reasoning", "")).strip()
    except json.JSONDecodeError:
        prediction = "INVALID"
        reasoning = "JSON Parsing Error"

    if prediction == "INVALID" or prediction not in VALID_EMOTIONS:
        prediction = normalize_prediction(prediction)

    return reasoning, prediction

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

def run_vllm_zero_shot(dataframe, output_file, window_sizes, max_dialogues=None):
    print("Loading model into VRAM...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ", 
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True
    )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    dialogue_ids = sorted(dataframe["Dialogue_ID"].unique())
    if max_dialogues is not None:
        dialogue_ids = dialogue_ids[:max_dialogues]

    df_filtered = dataframe[dataframe["Dialogue_ID"].isin(dialogue_ids)]

    prompts = []
    metadata = []

    print("Building prompts...")

    system_instruction = (
        "You are given structured dialogue data.\n\n"
        "Classify the emotion of the target utterance.\n\n"
        "IMPORTANT:\n"
        "- Use the dialogue context only if it helps interpret the target utterance\n"
        "- Focus on the emotional meaning of the target utterance\n"
        "- Choose exactly one emotion from the provided emotion options\n"
        "- Return ONLY valid JSON\n"
        "- Use DOUBLE quotes\n"
        "- Do not return markdown\n"
        "- Do not return any extra text outside the JSON object"
    )

    user_template = (
        "Input:\n"
        "{structured_input}\n\n"
        "Output format:\n"
        "{{\n"
        '    "reasoning": "...",\n'
        '    "emotion": "..."\n'
        "}}"
    )

    for dialogue_id, dialog in df_filtered.groupby("Dialogue_ID"):
        for window_size in window_sizes:
            window_df = dialog.tail(window_size).reset_index(drop=True)
            
            if window_df.empty:
                continue

            context_turns = []
            for _, row in window_df.iloc[:-1].iterrows():
                context_turns.append({
                    "speaker": row["Speaker"],
                    "utterance": row["Utterance"]
                })

            target_row = window_df.iloc[-1]
            true_label = target_row["Emotion"]

            structured_input_dict = {
                "task": "Classify the emotion of the target utterance",
                "emotion_options": VALID_EMOTIONS,
                "context": context_turns,
                "target_utterance": {
                    "speaker": target_row["Speaker"],
                    "utterance": target_row["Utterance"]
                }
            }

            user_msg = user_template.format(
                structured_input=json.dumps(structured_input_dict, indent=2)
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_msg}
            ]
            
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
                "prompt_word_count": len(formatted_prompt.split()),
                "target_utterance": target_row["Utterance"]
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
            "prompt_type": "structured_zero_shot",
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
    df = prepare_dataframe()

    output_file = "results/qwen_zero_shot_structured_dailydialog_results.csv"
    window_sizes = [1, 3, 5, 7, 9, 11]

    run_vllm_zero_shot(
        dataframe=df,
        output_file=output_file,
        window_sizes=window_sizes,
        max_dialogues=None
    )


if __name__ == "__main__":
    main()