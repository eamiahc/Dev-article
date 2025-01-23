import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from src.preprocess import preprocess_data

# Chargement des données
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Prétraitement
max_words = 5000
max_len = 100
embedding_dim = 100

X_train, y_train, tokenizer = preprocess_data(train_data, max_words, max_len)
X_test, y_test, _ = preprocess_data(test_data, max_words, max_len)

# Modèle
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(16, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    Dropout(0.7),
    Bidirectional(LSTM(8, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    Dropout(0.7),
    Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Sauvegarde du modèle
model.save("models/best_model.h5")
