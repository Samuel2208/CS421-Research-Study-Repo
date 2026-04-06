from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATASET_DIR = REPO_DIR / "Dataset"

MELD_PATH = DATASET_DIR / "MELD.csv"
# Updated output filenames to reflect the full filtered dataset
FILTERED_PATH = DATASET_DIR / "MELD_filtered_dialogues.csv"
IDS_PATH = DATASET_DIR / "MELD_filtered_dialogue_ids.csv"

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
    print("Number of eligible dialogues (>= 12 utterances):", len(eligible_dialogues))

    df = df[df["Dialogue_ID"].isin(eligible_dialogues["Dialogue_ID"])].copy()

    df["turn_index"] = df.groupby("Dialogue_ID").cumcount()

    # Keep only rows 0 through 11 (the first 12 utterances).
    df = df[df["turn_index"] <= 11].copy()

    # Get target rows only (row index 11).
    target_rows = df[df["turn_index"] == 11].copy()

    # >3 words in the target utterance.
    target_rows["target_word_count"] = target_rows["Utterance"].apply(count_words)
    target_rows = target_rows[target_rows["target_word_count"] > 3].copy()

    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    target_rows = target_rows[target_rows["Emotion"].isin(emotions)].copy()

    print("\nAvailable target rows by emotion (All valid dialogues):")
    print(target_rows["Emotion"].value_counts())

    # Instead of sampling, we keep ALL dialogue IDs that passed the filters
    valid_dialogue_ids = target_rows["Dialogue_ID"].tolist()

    print(f"\nTotal valid dialogues found: {len(valid_dialogue_ids)}")

    # Create the final subset containing all utterances for the valid dialogues
    filtered_df = df[df["Dialogue_ID"].isin(valid_dialogue_ids)].copy()

    filtered_df.to_csv(FILTERED_PATH, index=False)
    print("\nSaved full filtered dataset to:")
    print(FILTERED_PATH)

    ids_df = (
        target_rows[["Dialogue_ID", "Emotion", "Utterance"]]
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

    print("\nFiltered subset shape:", filtered_df.shape)
    print("Number of unique dialogues in filtered subset:", filtered_df["Dialogue_ID"].nunique())

if __name__ == "__main__":
    main()