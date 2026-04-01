import pandas as pd
import os
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

N = 28
dialog_ids = df[dialog_id_col].unique()[:N]

results = []

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

        prompt = (
            "You are an emotion classification assistant.\n\n"
            "Your task is to classify the emotion of the target utterance in the dialogue below.\n\n"
            "Choose exactly one label from this list:\n"
            "anger, disgust, fear, joy, neutral, sadness, surprise\n\n"

            "Here are some examples:\n\n"

            + "Example 1:\n"
            + "Dialogue:\n"
            + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_1)]) + "\n"
            + f"Target utterance:\n{example_1[-1]}\n"
            + f"Label: {example_1_emotion}\n\n"

            + "Example 2:\n"
            + "Dialogue:\n"
            + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_2)]) + "\n"
            + f"Target utterance:\n{example_2[-1]}\n"
            + f"Label: {example_2_emotion}\n\n"

            + "Now, classify the following:\n\n"
            + "Dialogue:\n"
            + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(context)]) + "\n\n"
            + f"Target utterance:\n{context[-1]}\n\n"

            "Explain your reasoning in one or two sentences behind your decision and then choose one label from the list above.\n\n"

            "Return your answer in this format:\n"
            "Reasoning: <your reasoning>\n"
            "Label: <one emotion label>"
        )

        print(f"\n--- Gemini Input for dialog {dialog_id}, context length {n} ---\n{prompt}\n")

        success = False
        while not success:
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt,
                )

                raw_output = response.text.strip() if response.text else ""

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

                accuracy = prediction == actual_emotion.lower()

                results.append(
                    {
                        "dialogue_id": dialog_id,
                        "window_size": n,
                        "prediction": prediction,
                        "label": actual_emotion,
                        "prompt_type": "few_shot_reverse_window",
                        "model": "gemini",
                        "reasoning": reasoning,
                        "accuracy": accuracy
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
     "prompt_type", "model", "reasoning", "accuracy"]
]

results_df.to_csv("gemini_meld_emotion_results.csv", index=False)