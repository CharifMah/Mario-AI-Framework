# data_utils.py

import os
import numpy as np
import json

HEIGHT = 16
WIDTH = 200

def load_levels(levels_path):
    levels = []
    for fname in os.listdir(levels_path):
        if fname.endswith('.txt'):
            with open(os.path.join(levels_path, fname), "r") as f:
                lines = [l.rstrip('\n') for l in f.readlines()]
                if len(lines) != HEIGHT:
                    continue  # skip wrong shape
                lines = [line.ljust(WIDTH, '-')[:WIDTH] for line in lines]
                levels.append(lines)
    return levels

def build_vocabulary(levels):
    chars = set()
    for level in levels:
        for row in level:
            chars.update(row)
    vocab = sorted(list(chars))
    char_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_char = {i: c for i, c in enumerate(vocab)}
    return vocab, char_to_int, int_to_char

def encode_levels(levels, char_to_int):
    encoded = []
    for level in levels:
        arr = np.zeros((HEIGHT, WIDTH), dtype=int)
        for i, row in enumerate(level):
            for j, c in enumerate(row):
                arr[i, j] = char_to_int.get(c, 0)
        encoded.append(arr)
    return np.array(encoded)

def save_mapping(char_to_int, int_to_char, path):
    with open(path, "w") as f:
        json.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char}, f)
