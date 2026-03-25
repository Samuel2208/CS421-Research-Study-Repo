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

# Load the MELD CSV file.
def load_meld(path=MELD_PATH):
    return pd.read_csv(path)


# Fix broken text encodings in MELD utterances.
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


def main():
    df = load_meld()

    # Keep MELD sorted correctly.
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    # Clean utterance text.
    df["Utterance"] = df["Utterance"].apply(clean_text)

    # Count how many utterances are in each dialogue.
    dialogue_counts = df.groupby("Dialogue_ID").size().reset_index(name="num_utterances")

    # Keep only dialogues that have at least 11 utterances.
    eligible_dialogues = dialogue_counts[dialogue_counts["num_utterances"] >= 11]

    print("Number of eligible dialogues:", len(eligible_dialogues))

    # Randomly sample 28 dialogue IDs with a fixed seed for reproducibility.
    sampled_dialogue_ids = eligible_dialogues["Dialogue_ID"].sample(n=28, random_state=42).tolist()

    print("\nSampled dialogue IDs:")
    print(sorted(sampled_dialogue_ids))

    # Keep only rows from those sampled dialogues.
    subset_df = df[df["Dialogue_ID"].isin(sampled_dialogue_ids)].copy()

    # Save the subset CSV.
    subset_df.to_csv(SUBSET_PATH, index=False)
    print("\nSaved subset to:")
    print(SUBSET_PATH)

    # Save just the selected dialogue IDs too.
    ids_df = pd.DataFrame({"dialogue_id": sorted(sampled_dialogue_ids)})
    ids_df.to_csv(IDS_PATH, index=False)

    print("\nSaved dialogue ID list to:")
    print(IDS_PATH)

    # Quick checks
    print("\nSubset shape:", subset_df.shape)
    print("Number of unique dialogues in subset:", subset_df["Dialogue_ID"].nunique())


if __name__ == "__main__":
    main()