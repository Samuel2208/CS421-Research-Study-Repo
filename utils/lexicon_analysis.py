import os
import csv
import string
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

emotion_lexicon = defaultdict(lambda: defaultdict(int))

seen_utterances = defaultdict(set)

def process_utterance(utterance: str, predicted_emotion: str, ignore_duplicates: bool = True): # Set ignore_duplicates to false if you dont want to remove dupe utterances
    if not isinstance(utterance, str) or not utterance.strip():
        return

    predicted_emotion = predicted_emotion.strip().lower()

    if len(predicted_emotion.split()) > 1:
        predicted_emotion = "other"

    tracking_string = utterance.strip().lower()
    
    if ignore_duplicates:
        if tracking_string in seen_utterances[predicted_emotion]:
            return
        else:
            seen_utterances[predicted_emotion].add(tracking_string)

    utterance = utterance.lower()
    utterance = utterance.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(utterance)

    for word in tokens:
        if word not in stop_words and word.strip():
            lemma = lemmatizer.lemmatize(word)
            emotion_lexicon[predicted_emotion][lemma] += 1

def export_lexicons(title: str):
    if not emotion_lexicon:
        print("Lexicon is empty. Nothing to export.")
        return

    output_dir = "lexicon_analysis"
    os.makedirs(output_dir, exist_ok=True)

    for emotion, word_counts in emotion_lexicon.items():
        safe_emotion = "".join([c for c in emotion if c.isalnum() or c == '_'])
        filename = os.path.join(output_dir, f"{title}_{safe_emotion}_lexicon_dictionary.csv")
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Word", "Count"])
            
            sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
            
            for word, count in sorted_words:
                writer.writerow([word, count])
                
        print(f"Exported: {filename}")