import pandas as pd
import os
import time
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

meld_path = "../../Dataset/MELD_28_dialogues.csv"
df = pd.read_csv(meld_path)

dialog_id_col = "Dialogue_ID"
utterance_col = "Utterance"
emotion_col = "Emotion"
utterance_index_col = "Utterance_ID"
speaker_col = "Speaker"

df = df.sort_values([dialog_id_col, utterance_index_col])

N = 56
dialog_ids = df[dialog_id_col].unique()[:N]

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

results = []

for dialog_id in dialog_ids:
    dialog = df[df[dialog_id_col] == dialog_id]

    utterances = dialog[utterance_col].tolist()
    speakers = dialog[speaker_col].tolist()
    actual_emotion = dialog[emotion_col].iloc[-1]

    max_len = min(11, len(utterances))

    for n in range(1, max_len + 1, 2):
        start_idx = len(utterances) - n
        window_speakers = speakers[start_idx:]
        window_utterances = utterances[start_idx:]
        
        context_turns = [
            {"speaker": spk, "utterance": utt} 
            for spk, utt in zip(window_speakers[:-1], window_utterances[:-1])
        ]
        
        structured_input_dict = {
            "task": "Classify the emotion of the target utterance",
            "emotion_options": VALID_EMOTIONS,
            "context": context_turns,
            "target_utterance": {
                "speaker": window_speakers[-1],
                "utterance": window_utterances[-1]
            }
        }
        
        prompt = user_template.format(
            structured_input=json.dumps(structured_input_dict, indent=2)
        )
        
        full_prompt_text = system_instruction + "\n" + prompt

        print(f"\n--- Gemini Input for dialog {dialog_id}, context length {n} ---\n{prompt}\n")

        success = False
        while not success:
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )

                raw_output = response.text.strip() if response.text else ""
                
                prediction, reasoning = parse_json_output(raw_output)
                accuracy = prediction == actual_emotion.lower()

                results.append(
                    {
                        "dialogue_id": dialog_id,
                        "window_size": n,
                        "prediction": prediction,
                        "label": actual_emotion,
                        "prompt_type": "few_shot_structured_json",
                        "model": "gemini",
                        "reasoning": reasoning,
                        "accuracy": accuracy,
                        "prompt_word_count": len(full_prompt_text.split()),
                        "prompt_character_count": len(full_prompt_text)
                    }
                )

                print(f"Dialog ID: {dialog_id}")
                print(f"Window size: {n}")
                print(f"Prediction: {prediction}")
                print(f"Reasoning: {reasoning}")
                print(f"Label: {actual_emotion}")
                print(f"Accuracy: {accuracy}\n")

                success = True

            except Exception as e:
                print(f"Error at dialog {dialog_id}, context length {n}: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)

        time.sleep(2)

results_df = pd.DataFrame(results)

results_df = results_df[
    ["dialogue_id", "window_size", "prediction", "label",
     "prompt_type", "model", "reasoning", "accuracy",
     "prompt_word_count", "prompt_character_count"] 
]

results_df.to_csv("gemini_meld_emotion_results_structured.csv", index=False)