import pandas as pd
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

meld_path = "../Dataset/MELD.csv"
df = pd.read_csv(meld_path)

dialog_id_col = "Dialogue_ID"
utterance_col = "Utterance"
emotion_col = "Emotion"
utterance_index_col = "Utterance_ID"

df = df.sort_values([dialog_id_col, utterance_index_col])

N = 2
dialog_ids = df[dialog_id_col].unique()[:N]

results = []

for dialog_id in dialog_ids:

    dialog = df[df[dialog_id_col] == dialog_id]

    utterances = dialog[utterance_col].tolist()
    actual_emotion = dialog[emotion_col].iloc[0]

    for n in range(1, min(13, len(utterances) + 1)):

        context = utterances[:n]

        example_1 = [
            "Hey, how are you?",
            "I'm good, thanks! How about you?",
            "Doing well, just a bit tired.",
        ]
        example_1_emotion = "neutral"

        example_2 = [
            "Did you hear what happened yesterday?",
            "No, what happened?",
            "It was unbelievable!",
        ]
        example_2_emotion = "surprise"

        prompt = (
            "Given the following dialogs and their emotions:\n"
            + "\n".join(
                [
                    f"Example 1:\n"
                    + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_1)])
                ]
            )
            + f"\nEmotion: {example_1_emotion}\n"
            + "\n".join(
                [
                    f"Example 2:\n"
                    + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_2)])
                ]
            )
            + f"\nEmotion: {example_2_emotion}\n"
            + "\nNow, given this dialog:\n"
            + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(context)])
            + "\nWhat is the emotion behind the first utterance? "
            "Describe it in only one word. Do not explain the reasoning. "
            "Select between neutral, surprise, fear, sadness, joy, disgust, and anger."
        )

        print(
            f"\n--- Gemini Input for dialog {dialog_id}, context length {n} ---\n{prompt}\n"
        )

        try:

            """
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=100
                )
            )

            gemini_emotion = response.text.strip() if response.text else ""
            """

            # placeholder for testing
            gemini_emotion = "neutral"

            results.append(
                {
                    "dialog_id": dialog_id,
                    "context_len": n,
                    "gemini_emotion": gemini_emotion,
                    "actual_emotion": actual_emotion,
                    "utterances": context,
                    "tested_utterance_idx": 0,
                    "tested_utterance": context[0] if context else "",
                }
            )

            print(f"Dialog ID: {dialog_id}")
            print(f"Context length: {n}")
            print(f"Gemini: {gemini_emotion}")
            print(f"Actual: {actual_emotion}\n")

        except Exception as e:
            print(f"Error at dialog {dialog_id}, context length {n}: {e}")

        time.sleep(2)

results_df = pd.DataFrame(results)
results_df.to_csv("gemini_meld_emotion_results.csv", index=False)