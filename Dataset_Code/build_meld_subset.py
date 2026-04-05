from pathlib import Path
import pandas as pd


# Base repo paths
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATASET_DIR = REPO_DIR / "Dataset"

MELD_PATH = DATASET_DIR / "MELD.csv"
SUBSET_PATH = DATASET_DIR / "MELD_28_dialogues.csv"
IDS_PATH = DATASET_DIR / "MELD_28_dialogue_ids.csv"

def load_meld(path=MELD_PATH):
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


def count_words(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())


def main():
    df = load_meld()

    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    df["Utterance"] = df["Utterance"].apply(clean_text)

    dialogue_counts = df.groupby("Dialogue_ID").size().reset_index(name="num_utterances")

    # Keep only dialogues that have at least 12 utterances.
    eligible_dialogues = dialogue_counts[dialogue_counts["num_utterances"] >= 12]

    print("Number of eligible dialogues:", len(eligible_dialogues))

    df = df[df["Dialogue_ID"].isin(eligible_dialogues["Dialogue_ID"])].copy()

    df["turn_index"] = df.groupby("Dialogue_ID").cumcount()

    # Keep only rows 0 through 11.
    df = df[df["turn_index"] <= 11].copy()

    # Get target rows only (row index 11).
    target_rows = df[df["turn_index"] == 11].copy()

    # >3 words in the target utterance.
    target_rows["target_word_count"] = target_rows["Utterance"].apply(count_words)
    target_rows = target_rows[target_rows["target_word_count"] > 3].copy()

    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    target_rows = target_rows[target_rows["Emotion"].isin(emotions)].copy()

    print("\nAvailable target rows by emotion:")
    print(target_rows["Emotion"].value_counts())

    # 4 of each emotion instead of randomly sampled entirely.
    sampled_dialogue_ids = (
        target_rows.groupby("Emotion", group_keys=False)
        .apply(lambda x: x.sample(n=8, random_state=42))
        ["Dialogue_ID"]
        .tolist()
    )

    print("\nSampled dialogue IDs:")
    print(sorted(sampled_dialogue_ids))

    subset_df = df[df["Dialogue_ID"].isin(sampled_dialogue_ids)].copy()

    subset_df.to_csv(SUBSET_PATH, index=False)
    print("\nSaved subset to:")
    print(SUBSET_PATH)

    ids_df = (
    target_rows[target_rows["Dialogue_ID"].isin(sampled_dialogue_ids)][["Dialogue_ID", "Emotion", "Utterance"]]
    .drop_duplicates()
    .sort_values("Dialogue_ID")
    .rename(columns={
        "Dialogue_ID": "dialogue_id",
        "Emotion": "target_emotion",
        "Utterance": "target_utterance"
    })
)

    ids_df.to_csv(IDS_PATH, index=False)

    print("\nSaved dialogue ID list to:")
    print(IDS_PATH)

    print("\nSubset shape:", subset_df.shape)
    print("Number of unique dialogues in subset:", subset_df["Dialogue_ID"].nunique())


if __name__ == "__main__":
    main()