import pandas as pd
from collections import defaultdict, Counter
import re
from nltk.corpus import stopwords

df = pd.read_csv("./results/least_to_most_results.csv")

VALID_EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

stop_words = set(stopwords.words("english"))

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()

    # remove stopwords + short words
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

word_emotion_counts = defaultdict(Counter)

for _, row in df.iterrows():
    words = tokenize(row["reasoning"])  # you can switch to utterance
    emotion = row["prediction"]

    if emotion == "unknown":
        continue  # skip parse errors

    for w in words:
        word_emotion_counts[w][emotion] += 1


print("\n========== TOP WORDS PER EMOTION ==========")

for emotion in VALID_EMOTIONS:
    print(f"\nTop words for {emotion}:")
    sorted_words = sorted(
        [(w, count[emotion]) for w, count in word_emotion_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    print(sorted_words[:10])


print("\n========== STRONGLY BIASED WORDS ==========")

for word, counts in word_emotion_counts.items():
    total = sum(counts.values())
    if total < 10:
        continue  # ignore rare words

    dominant_emotion = max(counts, key=counts.get)
    ratio = counts[dominant_emotion] / total

    if ratio > 0.8:
        print(f"{word} → {dominant_emotion} ({ratio:.2f})")


def get_positions(text):
    words = tokenize(text)
    n = len(words)

    return {
        "start": words[:2],
        "middle": words[n//2-1:n//2+1] if n > 2 else [],
        "end": words[-2:]
    }

position_counts = {
    "start": defaultdict(Counter),
    "middle": defaultdict(Counter),
    "end": defaultdict(Counter)
}

for _, row in df.iterrows():
    if row["prediction"] == "unknown":
        continue

    positions = get_positions(row["reasoning"])
    emotion = row["prediction"]

    for pos in positions:
        for w in positions[pos]:
            position_counts[pos][w][emotion] += 1


print("\n========== POSITIONAL WORD INSIGHTS ==========")

for pos in ["start", "middle", "end"]:
    print(f"\nTop words in {pos}:")
    sorted_words = sorted(
        [(w, sum(counts.values())) for w, counts in position_counts[pos].items()],
        key=lambda x: x[1],
        reverse=True
    )
    print(sorted_words[:10])


def tokenize_utterance(text):
    return tokenize(text)

utterance_counts = defaultdict(Counter)

for _, row in df.iterrows():
    if row["prediction"] == "unknown":
        continue

    # If you stored input_text, use last line
    if "input_text" in df.columns:
        last_utterance = row["input_text"].split("\n")[-1]
    else:
        continue

    words = tokenize_utterance(last_utterance)
    emotion = row["prediction"]

    for w in words:
        utterance_counts[w][emotion] += 1


print("\n========== UTTERANCE-LEVEL WORD INSIGHTS ==========")

for emotion in VALID_EMOTIONS:
    print(f"\nTop utterance words for {emotion}:")
    sorted_words = sorted(
        [(w, count[emotion]) for w, count in utterance_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    print(sorted_words[:10])


total_words = len(word_emotion_counts)
print(f"\nTotal unique words analyzed: {total_words}")

total_samples = len(df)
parse_errors = (df["prediction"] == "unknown").sum()
print(f"Parse errors: {parse_errors}/{total_samples}")