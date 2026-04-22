import pandas as pd
import re
from vllm import LLM, SamplingParams
import os
import sys
from pathlib import Path
import spacy

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
sys.path.append(str(REPO_DIR))
from utils import lexicon_analysis

def parse_text_output(raw_pred):
    """Extracts reasoning and emotion from plain text output."""
    reasoning = ""
    prediction = ""

    reasoning_match = re.search(r"Reasoning:\s*(.*)", raw_pred, re.IGNORECASE)
    label_match = re.search(r"Label:\s*(.*)", raw_pred, re.IGNORECASE)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    if label_match:
        prediction = label_match.group(1).strip().lower()

    if prediction == "":
        # Fallback if the model didn't format it perfectly
        prediction = raw_pred.lower()

    return prediction, reasoning

def main():
    print("Loading model into VRAM...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ", 
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True
    )
    tokenizer = llm.get_tokenizer()
    
    # 256 is usually enough for a couple of sentences of reasoning + label
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # Load Dataset
    print("Loading dataset...")
    meld_path = "../../Dataset/MELD_filtered_dialogues.csv"
    df = pd.read_csv(meld_path)

    dialog_id_col = "Dialogue_ID"
    utterance_col = "Utterance"
    emotion_col = "Emotion"
    utterance_index_col = "Utterance_ID"
    speaker_col = "Speaker"

    print("Lemmatizing utterances...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) 
    df[utterance_col] = df[utterance_col].astype(str).apply(
        lambda text: " ".join([token.lemma_ for token in nlp(text)])
    )

    df = df.sort_values([dialog_id_col, utterance_index_col])

    N = 268
    dialog_ids = df[dialog_id_col].unique()[:N]
    
    df_filtered = df[df[dialog_id_col].isin(dialog_ids)]

    prompts = []
    metadata = []

    print("Building prompts...")

    system_instruction = (
        "You are an emotion classification assistant.\n"
        "Your task is to classify the emotion of the target utterance in the dialogue below.\n"
        "Choose exactly one label from this list: anger, disgust, fear, joy, neutral, sadness, surprise.\n\n"
        "Here are some examples:\n\n"
        "Example 1:\n"
        "Dialogue:\n"
        "1: A: Hey, how are you?\n"
        "2: B: I'm good, thanks! How about you?\n"
        "3: A: Doing well, just a bit tired.\n"
        "Target utterance:\nA: Doing well, just a bit tired.\n"
        "Label: neutral\n\n"
        "Example 2:\n"
        "Dialogue:\n"
        "1: A: Did you hear what happened yesterday?\n"
        "2: B: No, what happened?\n"
        "3: A: It was unbelievable!\n"
        "Target utterance:\nA: It was unbelievable!\n"
        "Label: surprise\n"
    )

    user_template = (
        "Now, classify the following:\n\n"
        "Dialogue:\n"
        "{dialogue_context}\n\n"
        "Target utterance:\n{target_utterance}\n\n"
        "Explain your reasoning in one or two sentences behind your decision and then choose one label from the list above.\n\n"
        "Return your answer in this exact format:\n"
        "Reasoning: <your reasoning>\n"
        "Label: <one emotion label>"
    )

    for dialog_id, dialog in df_filtered.groupby(dialog_id_col):
        utterances = dialog[utterance_col].tolist()
        speakers = dialog[speaker_col].tolist()
        actual_emotion = dialog[emotion_col].iloc[-1]

        combined = [f"{spk}: {utt}" for spk, utt in zip(speakers, utterances)]
        max_len = min(11, len(combined))

        for n in range(1, max_len + 1, 2):
            context = combined[-n:]
            
            dialogue_text = "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(context)])
            target_text = context[-1]
            
            user_msg = user_template.format(
                dialogue_context=dialogue_text,
                target_utterance=target_text
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
                "dialogue_id": dialog_id,
                "window_size": n,
                "label": actual_emotion,
                "prompt_character_count": len(formatted_prompt),
                "prompt_word_count": len(formatted_prompt.split()),
                "target_utterance": target_text
            })

    print(f"Running inference on {len(prompts)} prompts simultaneously...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    print("Parsing results...")
    
    for i, output in enumerate(outputs):
        raw_output = output.outputs[0].text.strip()
        meta = metadata[i]

        prediction, reasoning = parse_text_output(raw_output)
        accuracy = prediction == meta["label"].lower()

        results.append({
            "dialogue_id": meta["dialogue_id"],
            "window_size": meta["window_size"],
            "prediction": prediction,
            "label": meta["label"],
            "prompt_type": "few_shot_reverse_window",
            "model": "qwen2.5-7b-instruct-awq",
            "reasoning": reasoning,
            "accuracy": accuracy,
            "prompt_word_count": meta["prompt_word_count"],
            "prompt_character_count": meta["prompt_character_count"]
        })
        # lexicon_analysis.process_utterance(meta["target_utterance"], prediction)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["dialogue_id", "window_size", "prediction", "label",
         "prompt_type", "model", "reasoning", "accuracy",
         "prompt_word_count", "prompt_character_count"] 
    ]

    output_filename = "qwen_2_shot_lemmatized_results.csv"
    results_df.to_csv(output_filename, index=False)
    # lexicon_analysis.export_lexicons("qwen_2_shot")
    print(f"Finished! Results saved to {output_filename}")

if __name__ == "__main__":
    main()