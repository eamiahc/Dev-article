from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def build_bilstm_model(vocab_size, embedding_dim, max_len, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                  output_dim=embedding_dim, 
                  weights=[embedding_matrix], 
                  input_length=max_len, 
                  trainable=False),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Charger les données prétraitées
    df = pd.read_csv("data/processed/cleaned_reviews_with_sentiment.csv")
    max_len = 500

    # Préparer les embeddings
    from gensim.models import Word2Vec
    sentences = df['Filtered_Review'].apply(eval).tolist()
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    embedding_matrix = np.zeros((len(word2vec_model.wv) + 1, 100))
    word_index = {word: i+1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
    for word, i in word_index.items():
        embedding_matrix[i] = word2vec_model.wv[word]

    # Préparer les séquences
    df['Sequences'] = df['Filtered_Review'].apply(
        lambda x: [word_index[word] for word in eval(x) if word in word_index]
    )
    X = pad_sequences(df['Sequences'], maxlen=max_len, padding='post')
    y = np.where(df['Rating'] > 2, 1, 0)

    # Diviser les données
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construire et entraîner le modèle
    model = build_bilstm_model(len(embedding_matrix), 100, max_len, embedding_matrix)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Sauvegarder le modèle
    model.save("results/models/bilstm_model.h5")
    print("Modèle BiLSTM sauvegardé.")
