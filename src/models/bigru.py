from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout

def build_bigru_model(vocab_size, embedding_dim, max_len, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                  output_dim=embedding_dim, 
                  weights=[embedding_matrix], 
                  input_length=max_len, 
                  trainable=False),
        Bidirectional(GRU(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(GRU(64)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
