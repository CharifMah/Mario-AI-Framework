#!/usr/bin/env python3
# models/gan_lsi/data_utils.py

import json
import glob
import os
import numpy as np
import tensorflow as tf

def load_index2str(path):
    """
    Charge index2str.json et renvoie un dict char→int.
    """
    with open(path, 'r', encoding='utf-8') as f:
        idx2str = json.load(f)
    return {v: int(k) for k, v in idx2str.items()}


def txt_to_onehot(path, rev_map, n_symbols, W=256):
    """
    Lit un .txt Mario et renvoie un array one-hot shape (H, W, n_symbols).
    
    - Tronque ou complète chaque ligne à largeur W.
    - rev_map: mapping char→indice.
    - n_symbols: taille de l’alphabet.
    """
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
    H = len(lines)
    onehot = np.zeros((H, W, n_symbols), dtype='float32')
    for i, line in enumerate(lines):
        # Tronquer ou compléter
        if len(line) < W:
            line = line + '-' * (W - len(line))
        else:
            line = line[:W]
        for j, ch in enumerate(line):
            idx = rev_map.get(ch, rev_map.get('-', 0))
            onehot[i, j, idx] = 1.0
    return onehot


def make_dataset(txt_glob, rev_map, batch_size=16, W=256):
    """
    Construit un tf.data.Dataset de one-hot arrays uniforme.
    """
    files = glob.glob(txt_glob, recursive=True)
    n_symbols = len(rev_map)

    def gen():
        for p in files:
            yield txt_to_onehot(p, rev_map, n_symbols, W)

    # On prend la hauteur H du premier fichier
    sample = txt_to_onehot(files[0], rev_map, n_symbols, W)
    H = sample.shape[0]

    output_shape = (H, W, n_symbols)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=output_shape, dtype=tf.float32)
    )
    return ds.shuffle(len(files)).batch(batch_size).prefetch(2)
