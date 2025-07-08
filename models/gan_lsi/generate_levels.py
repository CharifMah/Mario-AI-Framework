#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import tensorflow as tf

# 1) Hyper-paramètres
LATENT_DIM = 32
N_GEN      = 5
OUT_DIR    = "generated_levels"

# 2) Création du dossier de sortie
os.makedirs(OUT_DIR, exist_ok=True)

# 3) Chargement du générateur
generator = tf.keras.models.load_model("gan_savedmodel")

# 4) Chargement du mapping char→int et inversion
with open("char_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
# mapping["int_to_char"] doit être un dict de str(int) -> str
int_to_char = { int(k): v for k, v in mapping["int_to_char"].items() }

# 5) Fonction de décodage
def decode_level(encoded: np.ndarray, int_to_char: dict) -> list[str]:
    """
    encoded: array shape (H, W) de valeurs entières [0..V-1]
    int_to_char: dict qui mappe chaque entier vers son caractère
    Retourne une liste de H chaînes de longueur W
    """
    H, W = encoded.shape
    lines: list[str] = []
    for i in range(H):
        # Remplace chaque indice par son caractère et concatène
        line = "".join(int_to_char[int(encoded[i, j])] for j in range(W))
        lines.append(line)
    return lines

# 6) Génération et écriture
for i in range(N_GEN):
    noise = tf.random.normal([1, LATENT_DIM])
    pred = generator(noise, training=False)[0].numpy()      # (H, W, V)
    level_encoded = np.argmax(pred, axis=-1)               # (H, W)
    level_txt = decode_level(level_encoded, int_to_char)   # list[str]

    out_path = os.path.join(OUT_DIR, f"level_{i}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in level_txt:
            f.write(line + "\n")

    print(f"Niveau généré : {out_path}")
