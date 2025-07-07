#!/usr/bin/env python3
# models/gan_lsi/gan/gan_generator.py

import os
import json
import numpy as np
import tensorflow as tf
from .model_keras import build_generator  # import relatif

# --------------------------------------------------
# Chargement du mapping index → caractère Mario
# --------------------------------------------------
HERE = os.path.dirname(__file__)
# index2str.json est maintenant à models/gan_lsi/index2str.json
INDEX2STR_PATH = os.path.abspath(
    os.path.join(HERE, '..', 'index2str.json')
)
with open(INDEX2STR_PATH, 'r', encoding='utf-8') as f:
    index2str = json.load(f)


def gan_generate(z: np.ndarray,
                 model_path: str,
                 latent_dim: int = 32,
                 out_shape=(16, 256, 1)) -> str:
    """
    Génère un niveau Mario au format JSON à partir d’un vecteur latent.
    """
    # 1) Reconstruire et charger le générateur
    G = build_generator(latent_dim=latent_dim, out_shape=out_shape)
    G.load_weights(model_path)

    # 2) Préparer le batch
    z_arr = np.atleast_2d(z).astype('float32')

    # 3) Générer le niveau
    fake = G.predict(z_arr, verbose=0)  # (batch, h, w, c)
    fake0 = fake[0]

    # 4) Conversion en indices
    if fake0.shape[-1] > 1:
        im = np.argmax(fake0, axis=-1)
    else:
        im = np.squeeze(fake0, axis=-1)
        im = ((im + 1) / 2 * (len(index2str) - 1)).astype(int)

    return json.dumps(im.tolist())


def write_level_txt(json_level: str, output_path: str):
    """
    Écrit un .txt pour Mario-AI-Framework à partir d’un JSON string.
    """
    arr = json.loads(json_level)
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in arr:
            line = ''.join(index2str[str(x)] for x in row)
            f.write(line + "\n")
