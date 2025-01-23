import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Nettoyage des textes
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prétraitement des données
def preprocess_data(file_path, output_path):
    df = pd.read_csv(file_path)

    # Nettoyer les avis
    df['Cleaned_Review'] = df['Review'].apply(clean_text)

    # Classifier les sentiments
    df['Sentiment'] = df['Rating'].apply(lambda x: "positive" if x > 2 else "negative")

    # Tokenisation
    df['Tokenized_Review'] = df['Cleaned_Review'].apply(word_tokenize)

    # Suppression des mots vides
    stop_words = set(stopwords.words('english'))
    df['Without_Stopwords'] = df['Tokenized_Review'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )

    # Suppression des mots à faible fréquence
    all_words = [word for tokens in df['Without_Stopwords'] for word in tokens]
    word_freq = Counter(all_words)
    frequent_words = {word for word, freq in word_freq.items() if freq >= 2}
    df['Filtered_Review'] = df['Without_Stopwords'].apply(
        lambda tokens: [word for word in tokens if word in frequent_words]
    )

    # Suppression des échantillons vides
    df['Is_Empty'] = df['Filtered_Review'].apply(lambda tokens: len(tokens) == 0)
    df = df[~df['Is_Empty']]

    # Sauvegarder le fichier nettoyé
    df.to_csv(output_path, index=False)
    print(f"Données prétraitées sauvegardées dans {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    preprocess_data("data/raw/tripadvisor_hotel_reviews.csv", "data/processed/cleaned_reviews_with_sentiment.csv")
