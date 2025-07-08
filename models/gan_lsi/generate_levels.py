import tensorflow as tf
import numpy as np
from data_utils import load_mapping, decode_level

LATENT_DIM = 32
HEIGHT = 16        # à adapter selon tes données
WIDTH = 200       # à adapter selon tes données
N_GEN = 5          # nombre de niveaux à générer

generator = tf.keras.models.load_model("gan_savedmodel")
char_to_int, int_to_char = load_mapping("char_mapping.json")

for i in range(N_GEN):
    noise = tf.random.normal([1, LATENT_DIM])
    pred = generator(noise, training=False)[0].numpy()
    # prend la classe max pour chaque case
    level_encoded = np.argmax(pred, axis=-1)
    level_txt = decode_level(level_encoded, int_to_char)
    with open(f"generated_level_{i}.txt", "w") as f:
        for line in level_txt:
            f.write(line + "\n")
    print(f"Niveau généré : generated_level_{i}.txt")
