import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import json
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import to_categorical

# === PARAMÈTRES ===
SEQUENCE_LENGTH = 100
EPOCHS = 20
BATCH_SIZE = 64
MODEL_PATH = "mario_lstm_savedmodel"
MAPPING_PATH = "char_mapping.json"
DEFAULT_LEVEL_PATH = "../../levels/train1/"

def load_levels(path):
    """Charge une carte unique ou toutes les cartes d'un dossier."""
    if os.path.isfile(path):
        with open(path, "r") as f:
            lines = [line.rstrip('\n') for line in f]
            level_text = ''.join(lines)
    elif os.path.isdir(path):
        level_text = ""
        for fname in os.listdir(path):
            if fname.endswith(".txt"):
                with open(os.path.join(path, fname), "r") as f:
                    lines = [line.rstrip('\n') for line in f]
                    level_text += ''.join(lines)
    else:
        raise ValueError("Le chemin fourni n'est ni un fichier ni un dossier existant.")
    return level_text

def prepare_sequences(level_text, sequence_length):
    chars = sorted(list(set(level_text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    n_vocab = len(chars)
    X, y = [], []
    for i in range(len(level_text) - sequence_length):
        seq_in = level_text[i:i + sequence_length]
        seq_out = level_text[i + sequence_length]
        X.append([char_to_int[char] for char in seq_in])
        y.append(char_to_int[seq_out])
    X = np.reshape(X, (len(X), sequence_length, 1))
    X = X / float(n_vocab)
    y = to_categorical(y, num_classes=n_vocab)
    return X, y, n_vocab, char_to_int, int_to_char

def build_model(sequence_length, n_vocab):
    model = Sequential()
    model.add(Input(shape=(sequence_length, 1)))  # Entrée : séquence de longueur sequence_length, 1 feature par pas de temps
    #model.add(LSTM(256, return_sequences=True))   # 1ère couche LSTM, retourne toute la séquence

    model.add(LSTM(64, return_sequences=False))   # 1ère couche LSTM, retourne toute la séquence
    model.add(Dropout(0.2))                       # Dropout pour éviter le surapprentissage
   # model.add(LSTM(256))                          # 2ème couche LSTM, retourne la dernière sortie seulement
    #model.add(Dropout(0.2))                       # Dropout
    model.add(Dense(n_vocab, activation='softmax'))  # Couche de sortie dense : une probabilité par tuile/caractère possible
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def main():
    # Utilise le chemin par défaut si aucun argument n'est donné
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = DEFAULT_LEVEL_PATH
        print(f"Aucun chemin fourni, utilisation du chemin par défaut : {DEFAULT_LEVEL_PATH}")
    print("Chargement des niveaux...")
    level_text = load_levels(path)
    print(f"Nombre de caractères total : {len(level_text)}")
    print("Préparation des séquences...")
    X, y, n_vocab, char_to_int, int_to_char = prepare_sequences(level_text, SEQUENCE_LENGTH)
    print(f"Nombre de séquences : {len(X)}")
    print("Construction du modèle...")
    model = build_model(SEQUENCE_LENGTH, n_vocab)
    print("Entraînement du modèle...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print(f"Sauvegarde du modèle au format SavedModel dans {MODEL_PATH} ...")
    model.save(MODEL_PATH)  # Sauvegarde au format SavedModel (dossier)
    print(f"Sauvegarde du mapping caractères <-> entiers dans {MAPPING_PATH} ...")
    with open(MAPPING_PATH, "w") as f:
        json.dump({"char_to_int": char_to_int, "int_to_char": int_to_char}, f)
    print("Terminé !")

if __name__ == "__main__":
    main()
