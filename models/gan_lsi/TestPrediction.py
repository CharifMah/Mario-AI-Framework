import tensorflow as tf
import numpy as np
import json
import logging
from datetime import datetime

# Logging avec horodatage pour chaque ligne
logging.basicConfig(
    filename="log_result.txt",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Charger le mapping int_to_char
with open('char_mapping.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
int_to_char = mapping.get('int_to_char', [])
if isinstance(int_to_char, dict):
    # Option sécurité pour compatibilité (rare)
    int_to_char = [int_to_char[str(i)] for i in range(len(int_to_char))]

# Charger le modèle SavedModel
model = tf.saved_model.load("gan_savedmodel")
infer = model.signatures["serving_default"]

# Création d’un vecteur seed pour test (adapte si besoin)
z = np.random.randn(1, 32).astype(np.float32)  # 32 = latent_dim GAN, sinon adapte !

# Prédiction
result = infer(tf.convert_to_tensor(z))
# Pour GAN keras.export, souvent 'output_0' dans result. Adapte la clé si besoin !
output = list(result.values())[0].numpy()      # Shape ex: (1, 16, 150, 11)
# Moyenne sur toute la map, pour avoir 1 proba par caractère (option)
output_mean = output.mean(axis=(1,2))[0]       # (11,)

# Affiche et log chaque caractère avec la date/heure
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for idx, p in enumerate(output_mean):
    percent = p * 100
    char = int_to_char[idx] if isinstance(int_to_char, list) else int_to_char[str(idx)]
    line = f"{now} {char} : {percent:.2f}%"
    print(line)
    logging.info(line)
