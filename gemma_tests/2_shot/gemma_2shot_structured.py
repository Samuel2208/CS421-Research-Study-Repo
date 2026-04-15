import pandas as pd
import re
import json
from vllm import LLM, SamplingParams

def parse_json_output(raw_pred):
    """Extracts reasoning and emotion from JSON output."""
    match = re.search(r'\{.*\}', raw_pred.strip(), re.DOTALL)
    json_str = match.group(0) if match else raw_pred.strip()

    try:
        data = json.loads(json_str)
        prediction = str(data.get("emotion", "")).strip().lower()
        reasoning = str(data.get("reasoning", "")).strip()
        return prediction, reasoning
    except json.JSONDecodeError:
        # Fallback if the model completely fails to produce valid JSON
        return raw_pred.strip().lower(), "JSON Parsing Error"

def main():
    print("Loading Gemma 4 into VRAM (16GB optimized)...")
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

    df = df.sort_values([dialog_id_col, utterance_index_col])

    N = 268
    dialog_ids = df[dialog_id_col].unique()[:N]
    
    df_filtered = df[df[dialog_id_col].isin(dialog_ids)]

    prompts = []
    metadata = []

    print("Building prompts...")

    VALID_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

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
        "- Do not return any extra text outside the JSON object\n\n"
        "Here are some examples:\n\n"
        "Example 1 Input:\n"
        "{\n"
        '  "task": "Classify the emotion of the target utterance",\n'
        f'  "emotion_options": {json.dumps(VALID_EMOTIONS)},\n'
        '  "context": [\n'
        '    {"speaker": "A", "utterance": "Hey, how are you?"},\n'
        '    {"speaker": "B", "utterance": "I\'m good, thanks! How about you?"}\n'
        '  ],\n'
        '  "target_utterance": {\n'
        '    "speaker": "A",\n'
        '    "utterance": "Doing well, just a bit tired."\n'
        '  }\n'
        "}\n\n"
        "Example 1 Output:\n"
        "{\n"
        '  "reasoning": "The speaker is providing a standard polite response without expressing strong positive or negative feelings.",\n'
        '  "emotion": "neutral"\n'
        "}\n\n"
        "Example 2 Input:\n"
        "{\n"
        '  "task": "Classify the emotion of the target utterance",\n'
        f'  "emotion_options": {json.dumps(VALID_EMOTIONS)},\n'
        '  "context": [\n'
        '    {"speaker": "A", "utterance": "Did you hear what happened yesterday?"},\n'
        '    {"speaker": "B", "utterance": "No, what happened?"}\n'
        '  ],\n'
        '  "target_utterance": {\n'
        '    "speaker": "A",\n'
        '    "utterance": "It was unbelievable!"\n'
        '  }\n'
        "}\n\n"
        "Example 2 Output:\n"
        "{\n"
        '  "reasoning": "The use of the word unbelievable and an exclamation point strongly indicates shock or astonishment.",\n'
        '  "emotion": "surprise"\n'
        "}"
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

    for dialog_id, dialog in df_filtered.groupby(dialog_id_col):
        utterances = dialog[utterance_col].tolist()
        speakers = dialog[speaker_col].tolist()
        actual_emotion = dialog[emotion_col].iloc[-1]

        max_len = min(11, len(utterances))

        for n in range(1, max_len + 1, 2):
            # Extract the correct window for speakers and utterances
            start_idx = len(utterances) - n
            window_speakers = speakers[start_idx:]
            window_utterances = utterances[start_idx:]
            
            # Format context (everything except the last item)
            context_turns = [
                {"speaker": spk, "utterance": utt} 
                for spk, utt in zip(window_speakers[:-1], window_utterances[:-1])
            ]
            
            # Format the target input block
            structured_input_dict = {
                "task": "Classify the emotion of the target utterance",
                "emotion_options": VALID_EMOTIONS,
                "context": context_turns,
                "target_utterance": {
                    "speaker": window_speakers[-1],
                    "utterance": window_utterances[-1]
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
                "dialogue_id": dialog_id,
                "window_size": n,
                "label": actual_emotion,
                "prompt_character_count": len(formatted_prompt),
                "prompt_word_count": len(formatted_prompt.split())
            })

    print(f"Running inference on {len(prompts)} prompts simultaneously...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    print("Parsing results...")
    
    for i, output in enumerate(outputs):
        raw_output = output.outputs[0].text.strip()
        meta = metadata[i]

        prediction, reasoning = parse_json_output(raw_output)
        accuracy = prediction == meta["label"].lower()

        results.append({
            "dialogue_id": meta["dialogue_id"],
            "window_size": meta["window_size"],
            "prediction": prediction,
            "label": meta["label"],
            "prompt_type": "few_shot_structured_json",
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

    output_filename = "gemma_2_shot_structred_results.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Finished! Results saved to {output_filename}")

if __name__ == "__main__":
    main()