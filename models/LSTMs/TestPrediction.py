import tensorflow as tf
import numpy as np
import json
import logging

# Configuration du logger pour inclure la date et l'heure
logging.basicConfig(
    filename="log_result.txt",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Charger le mapping depuis le fichier JSON
with open('char_mapping.json', 'r', encoding='utf-8') as f:
    char_mapping = json.load(f)
int_to_char = char_mapping['int_to_char']

# Charger le modèle et faire une prédiction
model = tf.saved_model.load("mario_lstm_savedmodel")
infer = model.signatures["serving_default"]
x = np.zeros((1, 100, 1), dtype=np.float32)  # ou ton vrai seed

result = infer(tf.convert_to_tensor(x))
output = result['output_0'].numpy()[0]  # shape: (11,)

np.set_printoptions(suppress=True, precision=6)

print("Prédiction caractère par caractère :")
for idx, prob in enumerate(output):
    char = int_to_char[str(idx)]
    percent = prob * 100
    line = f"{char} : {percent:.2f}%"
    print(line)
    logging.info(line)
