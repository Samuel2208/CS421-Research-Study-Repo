import pandas as pd
import os
from dotenv import load_dotenv
from google import genai
import time

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

iemocap_paths = [
    "../Dataset/iemocap_session1.csv",
    "../Dataset/iemocap_session2.csv",
    "../Dataset/iemocap_session3.csv",
    "../Dataset/iemocap_session4.csv",
    "../Dataset/iemocap_session5.csv",
]

df = pd.concat([pd.read_csv(path) for path in iemocap_paths], ignore_index=True)

turn_col = "TURN"
utterance_col = "UTTERANCE"
emotion_col = "EMOTION"

df["DIALOG_ID"] = df[turn_col].apply(lambda x: "_".join(x.split("_")[:2]))

df["UTT_INDEX"] = df[turn_col].apply(lambda x: x.split("_")[-1][1:])

df = df.sort_values(["DIALOG_ID", "UTT_INDEX"])

dialog_id_col = "DIALOG_ID"

abberivation_to_full = {
    "-": "-",
    "ang": "anger",
    "dis": "disgust",
    "exc": "excitement",
    "fea": "fear",
    "fru": "frustration",
    "hap": "happiness",
    "neu": "neutral",
    "oth": "other",
    "sad": "sadness",
    "sur": "surprise",
    "xxx": "xxx",
}

N = 2
dialog_ids = df[dialog_id_col].unique()[:N]

results = []

for dialog_id in dialog_ids:

    dialog = df[df[dialog_id_col] == dialog_id]

    utterances = dialog[utterance_col].tolist()
    actual_emotions = dialog[emotion_col].tolist()

    for n in range(1, min(13, len(utterances) + 1)):

        context = utterances[:n]
        context_actual_emotions = actual_emotions[:n]

        example_1 = [
            "Hey, how are you?",
            "I'm good, thanks! How about you?",
            "Doing well, just a bit tired.",
        ]
        example_1_emotions = ["neutral", "neutral", "neutral"]

        example_2 = [
            "Did you hear what happened yesterday?",
            "No, what happened?",
            "It was unbelievable!",
        ]
        example_2_emotions = ["surprise", "neutral", "surprise"]

        prompt = (
            "Given the following dialogs and their emotions:\n"
            + "\n".join(
                [
                    f"Example 1:\n"
                    + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_1)])
                    + "\nEmotions: "
                    + ", ".join(example_1_emotions)
                ]
            )
            + "\n"
            + "\n".join(
                [
                    f"Example 2:\n"
                    + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(example_2)])
                    + "\nEmotions: "
                    + ", ".join(example_2_emotions)
                ]
            )
            + "\nNow, given this dialog:\n"
            + "\n".join([f"{i+1}: {utt}" for i, utt in enumerate(context)])
            + "\nWhat are the emotions behind each utterance? "
            "Respond with a comma-separated list of one word per utterance, in order. "
            "Do not explain the reasoning. Select between -, anger, disgust, excitement, "
            "fear, frustration, happiness, neutral, other, sadness, surprise, and xxx."
        )

        print(
            f"\n--- Gemini Input for dialog {dialog_id}, context length {n} ---\n{prompt}\n"
        )

        try:

            """
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )

            gemini_emotions = (
                response.text.strip().split(",")
                if response.text
                else ["neutral"] * n
            )
            """

            # placeholder for testing
            gemini_emotions = ["neutral"] * n

            for idx, (utt, gem_em, act_em) in enumerate(
                zip(context, gemini_emotions, context_actual_emotions)
            ):

                results.append(
                    {
                        "dialog_id": dialog_id,
                        "context_len": n,
                        "utterance_idx": idx,
                        "utterance": utt,
                        "gemini_emotion": gem_em.strip(),
                        "actual_emotion": (
                            abberivation_to_full[act_em]
                            if abberivation_to_full.get(act_em)
                            else act_em
                        ),
                    }
                )

                print(f"Dialog ID: {dialog_id}")
                print(f"Context length: {n}")
                print(f"Utterance idx: {idx}")
                print(f"Gemini: {gem_em.strip()}")
                print(
                    f"Actual: {abberivation_to_full[act_em] if abberivation_to_full.get(act_em) else act_em}\n"
                )

        except Exception as e:
            print(f"Error at dialog {dialog_id}, context length {n}: {e}")

        time.sleep(2)

results_df = pd.DataFrame(results)
results_df.to_csv("gemini_iemocap_emotion_results_all.csv", index=False)
