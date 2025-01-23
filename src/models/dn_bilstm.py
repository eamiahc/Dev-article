from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate, Attention

def build_dn_bilstm_model(vocab_size, embedding_dim, max_len, embedding_matrix):
    input_text = Input(shape=(max_len,), name="Text_Input")
    embedding_text = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(input_text)

    # Encodeur BiLSTM
    encoder = Bidirectional(LSTM(128, return_sequences=True))(embedding_text)

    # Attention
    attention = Attention()([encoder, encoder])

    # Décodeur BiLSTM
    decoder = Bidirectional(LSTM(128, return_sequences=False))(attention)

    # Sortie
    outputs = Dense(1, activation='sigmoid')(decoder)

    model = Model(inputs=input_text, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
