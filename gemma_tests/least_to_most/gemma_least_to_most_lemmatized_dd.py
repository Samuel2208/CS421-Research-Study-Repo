import pandas as pd
import re
import json
from vllm import LLM, SamplingParams
import os
import sys
from pathlib import Path
import spacy

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
sys.path.append(str(REPO_DIR))
from utils import lexicon_analysis

def parse_llm_output(raw_pred):
    clean_pred = re.sub(r'```json|```', '', raw_pred).strip()
    try:
        data = json.loads(clean_pred)
        emotion = data.get("emotion", "unknown").lower()
        reasoning = data.get("reasoning", "")
        return emotion, reasoning
    except Exception:
        return "unknown", clean_pred

def main():
    print("Loading model into VRAM...")
    llm = LLM(
        model="google/gemma-4-E4B-it", 
        quantization="fp8",
        max_model_len=4096,                   
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,           
        limit_mm_per_prompt={"image": 0, "audio": 0}, 
        trust_remote_code=True
    )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    dd_path = "../../Dataset/DailyDialog_filtered_dialogues.csv"
    df = pd.read_csv(dd_path)

    dialog_id_col = "Dialogue_ID"
    utterance_col = "Utterance"
    emotion_col = "Emotion"
    utterance_index_col = "Utterance_ID"

    print("Lemmatizing utterances...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) 
    df[utterance_col] = df[utterance_col].astype(str).apply(
        lambda text: " ".join([token.lemma_ for token in nlp(text)])
    )

    df = df.sort_values([dialog_id_col, utterance_index_col])
    dialog_ids = df[dialog_id_col].unique()

    prompts = []
    metadata = []

    print("Building prompts...")
    
    system_instruction = (
        "You are an emotion classification assistant. "
        "Your task is to determine the emotion of the LAST utterance in the given dialogue."
    )

    user_instruction_template = (
        "Follow these steps internally:\n"
        "1. Summarize each utterance briefly.\n"
        "2. Describe how the emotions evolve.\n"
        "3. Analyze the final utterance in context.\n"
        "4. Choose the final emotion from:\n"
        "[neutral, joy, sadness, anger, fear, disgust, surprise]\n\n"
        "IMPORTANT:\n"
        "- Return ONLY valid JSON\n"
        "- Do NOT include markdown (no ``` or ```json)\n"
        "- Do NOT include any extra text\n"
        "- Use DOUBLE quotes (\") for all keys and values\n\n"
        "Output format:\n"
        "{{\n"
        "    \"reasoning\": \"brief explanation\",\n"
        "    \"emotion\": \"...\"\n"
        "}}\n\n"
        "Dialogue:\n"
        "{dialogue}"
    )

    for dialog_id in dialog_ids:
        dialog = df[df[dialog_id_col] == dialog_id]

        utterances = dialog[utterance_col].tolist()
        speakers = ["A" if i % 2 == 0 else "B" for i in range(len(utterances))]
        actual_emotion = dialog[emotion_col].iloc[-1]

        combined = [f"{spk}: {utt}" for spk, utt in zip(speakers, utterances)]
        max_len = min(11, len(combined))

        for n in range(1, max_len + 1, 2):
            context = combined[-n:]
            dialogue_text = "\n".join(context)
            
            user_msg = user_instruction_template.format(dialogue=dialogue_text)

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
                "dialogue_id": dialog_id,
                "window_size": n,
                "label": actual_emotion,
                "prompt_character_count": len(formatted_prompt),
                "prompt_word_count": len(formatted_prompt.split()),
                "target_utterance": utterances[-1]
            })

    print(f"Running inference on {len(prompts)} prompts simultaneously...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    print("Parsing results...")
    
    for i, output in enumerate(outputs):
        raw_output = output.outputs[0].text.strip()
        meta = metadata[i]

        prediction, reasoning = parse_llm_output(raw_output)

        accuracy = prediction == meta["label"].lower()

        results.append({
            "dialogue_id": meta["dialogue_id"],
            "window_size": meta["window_size"],
            "prediction": prediction,
            "label": meta["label"],
            "prompt_type": "least_to_most",
            "model": "gemma-4-E4B-it",
            "reasoning": reasoning,
            "accuracy": accuracy,
            "prompt_word_count": meta["prompt_word_count"],
            "prompt_character_count": meta["prompt_character_count"]
        })

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["dialogue_id", "window_size", "prediction", "label",
         "prompt_type", "model", "reasoning", "accuracy",
         "prompt_word_count", "prompt_character_count"] 
    ]

    output_filename = "gemma_least_to_most_lemmatized_dailydialog_results.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Finished! Results saved to {output_filename}")

if __name__ == "__main__":
    main()