import pandas as pd
import re
from vllm import LLM, SamplingParams

def main():
    print("Loading model into VRAM...")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ", 
        quantization="awq",
        max_model_len=4096
    )
    tokenizer = llm.get_tokenizer()
    
    # Use temperature 0.0 for classification tasks to get deterministic, accurate results
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # 2. Load Dataset
    meld_path = "../Dataset/MELD_filtered_dialogues.csv"
    df = pd.read_csv(meld_path)

    dialog_id_col = "Dialogue_ID"
    utterance_col = "Utterance"
    emotion_col = "Emotion"
    utterance_index_col = "Utterance_ID"
    speaker_col = "Speaker"

    df = df.sort_values([dialog_id_col, utterance_index_col])

    N = 268
    dialog_ids = df[dialog_id_col].unique()[:N]

    prompts = []
    metadata = []

    print("Building prompts...")
    for dialog_id in dialog_ids:
        dialog = df[df[dialog_id_col] == dialog_id]

        utterances = dialog[utterance_col].tolist()
        speakers = dialog[speaker_col].tolist()
        actual_emotion = dialog[emotion_col].iloc[-1]

        combined = [f"{spk}: {utt}" for spk, utt in zip(speakers, utterances)]
        max_len = min(11, len(combined))

        for n in range(1, max_len + 1, 2):
            context = combined[-n:]

            example_1 = [
                "A: Hey, how are you?",
                "B: I'm good, thanks! How about you?",
                "A: Doing well, just a bit tired.",
            ]
            example_1_emotion = "neutral"

            example_2 = [
                "A: Did you hear what happened yesterday?",
                "B: No, what happened?",
                "A: It was unbelievable!",
            ]
            example_2_emotion = "surprise"

            system_instruction = (
                "You are an emotion classification assistant.\n"
                "Your task is to classify the emotion of the target utterance in the dialogue below.\n"
                "Choose exactly one label from this list: anger, disgust, fear, joy, neutral, sadness, surprise."
            )

            user_instruction = (
                "Here are some examples:\n\n"
                "Example 1:\n"
                "Dialogue:\n"
                + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_1)]) + "\n"
                + f"Target utterance:\n{example_1[-1]}\n"
                + f"Label: {example_1_emotion}\n\n"
                
                "Example 2:\n"
                "Dialogue:\n"
                + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_2)]) + "\n"
                + f"Target utterance:\n{example_2[-1]}\n"
                + f"Label: {example_2_emotion}\n\n"
                
                "Now, classify the following:\n\n"
                "Dialogue:\n"
                + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(context)]) + "\n\n"
                + f"Target utterance:\n{context[-1]}\n\n"
                
                "Explain your reasoning in one or two sentences behind your decision and then choose one label from the list above.\n\n"
                "Return your answer in this format:\n"
                "Reasoning: <your reasoning>\n"
                "Label: <one emotion label>"
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
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

        reasoning = ""
        prediction = ""

        reasoning_match = re.search(r"Reasoning:\s*(.*)", raw_output, re.IGNORECASE)
        label_match = re.search(r"Label:\s*(.*)", raw_output, re.IGNORECASE)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        if label_match:
            prediction = label_match.group(1).strip().lower()

        if prediction == "":
            prediction = raw_output.lower()

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

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["dialogue_id", "window_size", "prediction", "label",
         "prompt_type", "model", "reasoning", "accuracy",
         "prompt_word_count", "prompt_character_count"] 
    ]

    output_filename = "qwen_meld_emotion_results.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Finished! Results saved to {output_filename}")

if __name__ == "__main__":
    main()