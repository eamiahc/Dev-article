import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(data, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['Cleaned_Review'])
    
    sequences = tokenizer.texts_to_sequences(data['Cleaned_Review'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    labels = data['Sentiment'].map({'Positive': 1, 'Negative': 0}).values
    
    return padded_sequences, labels, tokenizer
