from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATASET_DIR = REPO_DIR / "Dataset"

INPUT_PATH = DATASET_DIR / "DailyDialog.csv"
FILTERED_PATH = DATASET_DIR / "DailyDialog_filtered_dialogues.csv"
IDS_PATH = DATASET_DIR / "DailyDialog_filtered_dialogue_ids.csv"

# DailyDialog Emotion Mapping
EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "joy",       # Mapped from happiness to match MELD
    5: "sadness",
    6: "surprise"
}

def clean_text(text):
    if pd.isna(text):
        return text

    text = str(text)

    replacements = {
        " ' ": "'",
        " ,": ",",
        " .": ".",
        " !": "!",
        " ?": "?",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text.strip()

def count_words(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

def parse_emotions(emo_str):
    return [int(x) for x in re.findall(r'\d+', str(emo_str))]

def parse_dialog(dialog_str):
    if pd.isna(dialog_str): 
        return []
        
    c = str(dialog_str).strip()
    if c.startswith('['): c = c[1:]
    if c.endswith(']'): c = c[:-1]
    
    # Safely extract utterances ignoring internal apostrophes
    # Matches text enclosed in either "..." or '...'
    matches = re.finditer(r'\"([^\"]+)\"|\'([^\']+)\'', c)
    
    utterances = []
    for match in matches:
        # Group 1 is double-quoted, Group 2 is single-quoted
        utt = match.group(1) if match.group(1) is not None else match.group(2)
        if utt and utt.strip():
            utterances.append(utt.strip())
            
    return utterances

def unroll_dataset(df):
    rows = []
    for idx, row in df.iterrows():
        utts = parse_dialog(row['dialog'])
        emos = parse_emotions(row['emotion'])
        
        # Only process if arrays align
        if len(utts) == len(emos) and len(utts) > 0:
            for u_idx, (u, e) in enumerate(zip(utts, emos)):
                rows.append({
                    "Dialogue_ID": idx, # Automatically generated from the row number!
                    "Utterance_ID": u_idx,
                    "Utterance": clean_text(u),
                    "Emotion": EMOTION_MAP.get(e, "unknown")
                })
                
    return pd.DataFrame(rows)

def main():
    raw_df = pd.read_csv(INPUT_PATH)
    
    # Expand into the standard row-by-row format
    df = unroll_dataset(raw_df)

    # Get target rows only (the LAST utterance of each dialogue)
    target_rows = df.groupby("Dialogue_ID").tail(1).copy()

    # >2 words in the target utterance
    target_rows["target_word_count"] = target_rows["Utterance"].apply(count_words)
    target_rows = target_rows[target_rows["target_word_count"] > 2].copy()

    valid_emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    target_rows = target_rows[target_rows["Emotion"].isin(valid_emotions)].copy()

    print("\nAvailable target rows by emotion (All valid dialogues):")
    print(target_rows["Emotion"].value_counts())

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