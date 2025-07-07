import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json

LEVELS_DIR = "../../levels/hopper"
MODEL_PATH = "gan_savedmodel"      # C'est bien un dossier, PAS d'extension
HEIGHT = 16
WIDTH = 150
EPOCHS = 1000
BATCH_SIZE = 8
LATENT_DIM = 32

def load_levels_from_dir(levels_dir, height, width):
    levels = []
    charset = set()
    for fname in os.listdir(levels_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(levels_dir, fname), "r") as f:
                lines = [line.rstrip('\n') for line in f]
                if len(lines) < height:
                    continue
                lines = lines[:height]
                level = ''.join(lines)
                charset.update(level)
                levels.append(level)
    return levels, sorted(list(charset))

def encode_levels(levels, charset, height, width):
    char2idx = {c: i for i, c in enumerate(charset)}
    n_symbols = len(charset)
    X = np.zeros((len(levels), height, width, n_symbols), dtype=np.float32)
    for i, level in enumerate(levels):
        for y in range(height):
            for x in range(width):
                idx = char2idx[level[y*width + x]]
                X[i, y, x, idx] = 1.0
    return X, char2idx

def build_generator(latent_dim, height, width, n_symbols):
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(height * width * n_symbols, activation="relu"),
        layers.Reshape((height, width, n_symbols)),
        layers.Activation("softmax"),
    ], name="Generator")
    return model

def build_discriminator(height, width, n_symbols):
    model = tf.keras.Sequential([
        layers.Input(shape=(height, width, n_symbols)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ], name="Discriminator")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")
    return model

def train():
    levels, int_to_char = load_levels_from_dir(LEVELS_DIR, HEIGHT, WIDTH)
    print(f"{len(levels)} niveaux trouvés, {len(int_to_char)} symboles")
    X, char_to_int = encode_levels(levels, int_to_char, HEIGHT, WIDTH)
    n_symbols = len(int_to_char)

    generator = build_generator(LATENT_DIM, HEIGHT, WIDTH, n_symbols)
    discriminator = build_discriminator(HEIGHT, WIDTH, n_symbols)
    discriminator.trainable = False

    z = layers.Input(shape=(LATENT_DIM,))
    fake = generator(z)
    validity = discriminator(fake)
    gan = tf.keras.Model(z, validity, name="GAN")
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")

    for epoch in range(1, EPOCHS + 1):
        idx = np.random.randint(0, X.shape[0], BATCH_SIZE)
        real = X[idx]
        noise = np.random.randn(BATCH_SIZE, LATENT_DIM).astype("float32")
        gen_levels = generator.predict(noise, verbose=0)
        valid = np.ones((BATCH_SIZE, 1))
        fake_label = np.zeros((BATCH_SIZE, 1))

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real, valid)
        d_loss_fake = discriminator.train_on_batch(gen_levels, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        discriminator.trainable = False

        noise = np.random.randn(BATCH_SIZE, LATENT_DIM).astype("float32")
        g_loss = gan.train_on_batch(noise, valid)

        print(f"Epoch {epoch}/{EPOCHS} | d_loss={d_loss:.3f} | g_loss={g_loss:.3f}")

    # **Sauvegarde au format SavedModel avec export**
    generator.export(MODEL_PATH)
    print(f"Modèle générateur exporté au format SavedModel dans {MODEL_PATH}")

    with open("char_mapping.json", "w") as f:
        json.dump({"char_to_int": char_to_int, "int_to_char": int_to_char}, f)
    print("Mapping char_to_int/int_to_char sauvegardé dans char_mapping.json")

if __name__ == "__main__":
    train()
