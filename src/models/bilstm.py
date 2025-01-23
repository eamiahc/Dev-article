from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

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
