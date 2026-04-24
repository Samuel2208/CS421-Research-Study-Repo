import pandas as pd
from collections import defaultdict, Counter
import re

VALID_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

stop_words = {
    "the","is","and","a","to","of","in","that","it","on","for",
    "with","as","was","but","be","at","by","an","this","are",
    "you","i","me","my","we","our","your","he","she","they",
    "yeah","just","like","really","okay","ok","uh","um"
}

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

def clean_utterance(line):
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.strip()

def get_last_utterance(row):
    if "input_text" not in row or not isinstance(row["input_text"], str):
        return None
    last_line = row["input_text"].split("\n")[-1]
    return clean_utterance(last_line)

def build_lexicons(train_df):
    true_lexicon = defaultdict(Counter)
    pred_lexicon = defaultdict(Counter)

    for _, row in train_df.iterrows():
        utterance = get_last_utterance(row)
        if not utterance:
            continue

        words = tokenize(utterance)

        true_label = str(row["label"]).lower()
        pred_label = str(row["prediction"]).lower()

        for w in words:
            true_lexicon[true_label][w] += 1
            pred_lexicon[pred_label][w] += 1

    return true_lexicon, pred_lexicon

def print_top_words(lexicon, title):
    print(f"\n===== {title} =====")
    for emotion in VALID_EMOTIONS:
        print(f"\nTop words for {emotion}:")
        sorted_words = sorted(
            lexicon[emotion].items(),
            key=lambda x: x[1],
            reverse=True
        )
        print(sorted_words[:10])

def compare_words(true_lexicon, pred_lexicon, test_words=None):
    print("\n===== WORD COMPARISON =====")
    if test_words is None:
        overall_counts = Counter()
        for emotion in true_lexicon:
            overall_counts.update(true_lexicon[emotion])

    
        test_words = [w for w, _ in overall_counts.most_common(15)]
    


    print("\nAuto-selected words:", test_words)

    for word in test_words:
        print(f"\nWord: {word}")
        print("True:", {e: true_lexicon[e][word] for e in VALID_EMOTIONS})
        print("Pred:", {e: pred_lexicon[e][word] for e in VALID_EMOTIONS})

def lexicon_predict(words, lexicon):
    scores = defaultdict(int)

    for w in words:
        for emotion in lexicon:
            scores[emotion] += lexicon[emotion].get(w, 0)

    if all(v == 0 for v in scores.values()):
        return "unknown"

    return max(scores, key=scores.get)

def evaluate(test_df, true_lexicon):
    correct = 0
    total = 0
    unknown_count = 0

    for _, row in test_df.iterrows():
        utterance = get_last_utterance(row)
        if not utterance:
            continue

        words = tokenize(utterance)

        pred = lexicon_predict(words, true_lexicon)
        true = str(row["label"]).lower()

        if pred == "unknown":
            unknown_count += 1

        if pred == true:
            correct += 1

        total += 1

    if total > 0:
        accuracy = correct / total
        unknown_rate = unknown_count / total

        print("\n===== LEXICON MODEL RESULTS =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Unknown rate: {unknown_rate:.4f}")
        print(f"Coverage: {(1 - unknown_rate):.4f}")

def main():
    print("Loading data...")
    df = pd.read_csv("./results/least_to_most_results.csv")

    df = df[df["prediction"] != "unknown"].copy()

    train_df = df.sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    print("\nBuilding lexicons...")
    true_lexicon, pred_lexicon = build_lexicons(train_df)

    print_top_words(true_lexicon, "TRUE LEXICON")
    print_top_words(pred_lexicon, "PREDICTED LEXICON")

    # test_words = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]

    compare_words(true_lexicon, pred_lexicon)

    evaluate(test_df, true_lexicon)

if __name__ == "__main__":
    main()