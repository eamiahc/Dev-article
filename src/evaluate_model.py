import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd
from src.preprocess import preprocess_data

# Charger les données de test et le modèle
test_data = pd.read_csv("data/test.csv")
model = tf.keras.models.load_model("models/best_model.h5")

# Prétraitement des données
max_words = 5000
max_len = 100
X_test, y_test, _ = preprocess_data(test_data, max_words, max_len)

# Évaluation
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))
