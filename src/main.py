from src.utils.data_preprocessing import preprocess_data
from src.models.bilstm import build_bilstm_model
from src.models.text_cnn import build_text_cnn_model
from src.models.bigru import build_bigru_model
from src.models.dn_bilstm import build_dn_bilstm_model
import tensorflow as tf

if __name__ == "__main__":
    # Prétraiter les données
    preprocess_data("data/raw/tripadvisor_hotel_reviews.csv", "data/processed/cleaned_reviews_with_sentiment.csv")

    # Charger les données prétraitées
    import pandas as pd
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    from gensim.models import Word2Vec

    df = pd.read_csv("data/processed/cleaned_reviews_with_sentiment.csv")
    max_len = 500

    sentences = df['Filtered_Review'].apply(eval).tolist()
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    embedding_matrix = np.zeros((len(word2vec_model.wv) + 1, 100))
    word_index = {word: i+1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
    for word, i in word_index.items():
        embedding_matrix[i] = word2vec_model.wv[word]

    df['Sequences'] = df['Filtered_Review'].apply(
        lambda x: [word_index[word] for word in eval(x) if word in word_index]
    )
    X = pad_sequences(df['Sequences'], maxlen=max_len, padding='post')
    y = np.where(df['Rating'] > 2, 1, 0)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner et évaluer chaque modèle
    print("Training BiLSTM model...")
    bilstm_model = build_bilstm_model(len(embedding_matrix), 100, max_len, embedding_matrix)
    bilstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    print("Training Text-CNN model...")
    text_cnn_model = build_text_cnn_model(len(embedding_matrix), 100, max_len, embedding_matrix)
    text_cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    print("Training BiGRU model...")
    bigru_model = build_bigru_model(len(embedding_matrix), 100, max_len, embedding_matrix)
    bigru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    print("Training DN-BiLSTM model...")
    dn_bilstm_model = build_dn_bilstm_model(len(embedding_matrix), 100, max_len, embedding_matrix)
    dn_bilstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    print("All models trained successfully.")
