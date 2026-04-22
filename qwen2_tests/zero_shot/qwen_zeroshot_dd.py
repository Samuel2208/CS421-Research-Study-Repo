from pathlib import Path
import os
import pandas as pd
import re
from vllm import LLM, SamplingParams
import sys

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
sys.path.append(str(REPO_DIR))
from utils import lexicon_analysis

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
DATASET_DIR = REPO_DIR / "Dataset"
SUBSET_PATH = DATASET_DIR / "DailyDialog_filtered_dialogues.csv"

def load_dailydialog_subset(path=SUBSET_PATH):
    return pd.read_csv(path)

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    replacements = {
        "Â¬Ãs": "'s", "Â¬Ãve": "'ve", "Â¬Ãll": "'ll", "Â¬Ãre": "'re",
        "Â¬Ãm": "'m", "Â¬Ãd": "'d", "Â¬Ã": "'", "¬ís": "'s",
        "¬íve": "'ve", "¬íll": "'ll", "¬íre": "'re", "¬ím": "'m",
        "¬íd": "'d", "¬í": "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def prepare_dailydialog_dataframe(path=SUBSET_PATH):
    df = load_dailydialog_subset(path)
    if "Speaker" not in df.columns:
        df["Speaker"] = df["Utterance_ID"].apply(lambda x: "A" if x % 2 == 0 else "B")
    
    df = df[["Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion"]].copy()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    df["Utterance"] = df["Utterance"].apply(clean_text)
    return df

def format_context_for_prompt(context_df):
    if context_df is None or context_df.empty:
        return ""
    lines = []
    for _, row in context_df.iterrows():
        speaker = row["Speaker"]
        utterance = row["Utterance"]
        lines.append(f"{speaker}: {utterance}")
    return "\n".join(lines)

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
    if prediction == "INVALID":
        prediction = normalize_prediction(response_text)
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
    system_instruction = (
        "You are an emotion classification assistant.\n\n"
        "Your task is to classify the emotion of the target utterance in the dialogue below.\n\n"
        "Choose exactly one label from this list:\n"
        "anger, disgust, fear, joy, neutral, sadness, surprise"
    )
    user_template = (
        "Dialogue:\n{context_text}\n\n"
        "Target utterance:\n{target_utterance}\n\n"
        "Explain your reasoning in one or two sentences behind your decision and then choose one label from the list above.\n\n"
        "Return your answer in this format:\n"
        "Reasoning: <your reasoning>\n"
        "Label: <one emotion label>"
    )
    for dialogue_id, dialog in df_filtered.groupby("Dialogue_ID"):
        for window_size in window_sizes:
            window_df = dialog.tail(window_size).reset_index(drop=True)
            if window_df.empty:
                continue
            formatted_context = format_context_for_prompt(window_df)
            target_utterance = window_df.iloc[-1]["Utterance"]
            true_label = window_df.iloc[-1]["Emotion"]
            user_msg = user_template.format(
                context_text=formatted_context,
                target_utterance=target_utterance
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
                "target_utterance": target_utterance
            })
    outputs = llm.generate(prompts, sampling_params)
    results = []
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
    df = prepare_dailydialog_dataframe()
    output_file = "results/qwen_zero_shot_dailydialog_results.csv"
    window_sizes = [1, 3, 5, 7, 9, 11]
    run_vllm_zero_shot(
        dataframe=df,
        output_file=output_file,
        window_sizes=window_sizes,
        max_dialogues=None
    )

if __name__ == "__main__":
    main()