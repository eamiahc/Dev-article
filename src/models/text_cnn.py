from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate

def build_text_cnn_model(vocab_size, embedding_dim, max_len, embedding_matrix):
    inputs = Input(shape=(max_len,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)(inputs)

    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(embedding)
    pool1 = GlobalMaxPooling1D()(conv1)

    conv2 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same')(embedding)
    pool2 = GlobalMaxPooling1D()(conv2)

    conv3 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(embedding)
    pool3 = GlobalMaxPooling1D()(conv3)

    concatenated = Concatenate()([pool1, pool2, pool3])
    x = Dropout(0.3)(concatenated)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
