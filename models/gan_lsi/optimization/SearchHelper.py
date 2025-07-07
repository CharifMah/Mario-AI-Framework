#!/usr/bin/env python3
# models/gan_lsi/optimization/SearchHelper.py

import os
import random
import json
import numpy as np
from numpy.linalg import eig

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


def get_char(x: int) -> str:
    """Retourne le caractère Mario pour l’indice x."""
    return index2str[str(x)]


def to_level(json_level: str) -> str:
    """
    Convertit un JSON string en .txt pour Mario-AI-Framework.
    """
    arr = json.loads(json_level)
    lines = []
    for row in arr:
        lines.append(''.join(get_char(int(x)) for x in row) + "\n")
    return ''.join(lines)


class Individual:
    """Représente un individu (vecteur latent + métriques)."""
    def __init__(self):
        self.ID = None
        self.param_vector = None
        self.features = None
        self.fitness = None
        self.statsList = None
        self.delta = None
        self.emitter_id = None


class DecompMatrix:
    """Décomposition de la matrice de covariance pour CMA-ES."""
    def __init__(self, dimension: int):
        self.C = np.eye(dimension, dtype=float)
        self.eigenbasis = np.eye(dimension, dtype=float)
        self.eigenvalues = np.ones(dimension, dtype=float)
        self.condition_number = 1.0
        self.invsqrt = np.eye(dimension, dtype=float)

    def update_eigensystem(self):
        for i in range(self.C.shape[0]):
            for j in range(i):
                self.C[i, j] = self.C[j, i]
        vals, vecs = eig(self.C)
        self.eigenvalues = np.real(vals)
        self.eigenbasis = np.real(vecs)
        self.condition_number = np.max(self.eigenvalues) / np.min(self.eigenvalues)
        for i in range(self.C.shape[0]):
            for j in range(i+1):
                s = 0.0
                for k in range(len(vals)):
                    s += (self.eigenbasis[i, k] *
                          self.eigenbasis[j, k] /
                          np.sqrt(self.eigenvalues[k]))
                self.invsqrt[i, j] = self.invsqrt[j, i] = s


class FeatureMap:
    """
    Archive MAP-Elites multidimensionnelle.
    - feature_ranges: liste de (min, max) pour chaque BC
    - resolutions: nombre de cases par dimension
    """
    def __init__(self, feature_ranges, resolutions):
        self.feature_ranges = feature_ranges
        self.resolutions = resolutions
        self.elite_map = {}
        self.elite_indices = []

    def get_feature_index(self, fid: int, feature: float) -> int:
        lo, hi = self.feature_ranges[fid]
        if feature <= lo:
            return 0
        if feature >= hi:
            return self.resolutions[fid] - 1
        pos = (feature - lo) / (hi - lo)
        return int(pos * (self.resolutions[fid] - 1))

    def get_index(self, ind: Individual) -> tuple:
        return tuple(
            self.get_feature_index(i, f)
            for i, f in enumerate(ind.features)
        )

    def add(self, ind: Individual) -> bool:
        idx = self.get_index(ind)
        current = self.elite_map.get(idx)
        if current is None:
            self.elite_map[idx] = ind
            self.elite_indices.append(idx)
            ind.delta = (1, ind.fitness)
            return True
        elif ind.fitness > current.fitness:
            ind.delta = (0, ind.fitness - current.fitness)
            self.elite_map[idx] = ind
            return True
        return False

    def get_random_elite(self) -> Individual:
        idx = random.choice(self.elite_indices)
        return self.elite_map[idx]

    def save(self, filename: str):
        elites = list(self.elite_map.values())
        np.save(filename, elites)
