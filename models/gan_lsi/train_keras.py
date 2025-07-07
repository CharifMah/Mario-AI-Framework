#!/usr/bin/env python3
# models/gan_lsi/train_keras.py

import os
import argparse
import numpy as np
import tensorflow as tf

from .gan.model_keras import build_generator, build_discriminator
from .data_utils      import load_index2str, make_dataset

def parse_args():
    p = argparse.ArgumentParser("Train GAN sur niveaux Mario")
    p.add_argument("--epochs",      type=int,   default=50,
                   help="Nombre d'époques d'entraînement")
    p.add_argument("--batch-size",  type=int,   default=16,
                   help="Taille des batchs")
    p.add_argument("--latent-dim",  type=int,   default=32,
                   help="Dimension du vecteur latent")
    p.add_argument("--orig-glob",   type=str,   default="levels/original/*.txt",
                   help="Glob pour charger les .txt originaux")
    p.add_argument("--width",       type=int,   default=256,
                   help="Largeur normalisée des niveaux")
    p.add_argument("--out-dir",     type=str,   default="models/gan_lsi/checkpoints",
                   help="Répertoire de sortie pour poids & modèles")
    return p.parse_args()

def train(epochs, batch_size, latent_dim, orig_glob, width, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Charge mapping et crée dataset
    idx2str_path = os.path.join(os.path.dirname(__file__), "index2str.json")
    rev_map   = load_index2str(idx2str_path)
    n_symbols = len(rev_map)
    dataset   = make_dataset(orig_glob, rev_map, batch_size, W=width)

    # 2) Déduit la hauteur H d’après un batch
    sample = next(iter(dataset))
    H, W, C = sample.shape[1:]  # (batch, H, W, C)

    # 3) Construit Generator + Discriminator
    G = build_generator(latent_dim=latent_dim,
                        out_shape=(H, W),
                        n_symbols=C)
    D = build_discriminator()

    # 4) Compile D
    D.compile(optimizer=tf.keras.optimizers.RMSprop(5e-5),
              loss="binary_crossentropy")

    # 5) Compose le GAN par API fonctionnelle
    z = tf.keras.Input(shape=(latent_dim,))
    fake = G(z)
    valid = D(fake)
    GAN = tf.keras.Model(z, valid, name="GAN")
    GAN.compile(optimizer=tf.keras.optimizers.RMSprop(5e-5),
                loss="binary_crossentropy")

    # 6) Boucle d’entraînement
    for epoch in range(1, epochs+1):
        for real in dataset:
            D.train_on_batch(real, np.ones((real.shape[0],1)))
            zf = np.random.randn(real.shape[0], latent_dim).astype("float32")
            fakeb = G.predict(zf, verbose=0)
            D.train_on_batch(fakeb, np.zeros((fakeb.shape[0],1)))
            z2 = np.random.randn(batch_size, latent_dim).astype("float32")
            GAN.train_on_batch(z2, np.ones((batch_size,1)))
        print(f"Epoch {epoch}/{epochs} terminé")

    # 7) Sauvegarde échantillon + poids
    sample_out = G.predict(np.random.randn(1, latent_dim).astype("float32"))
    np.save(os.path.join(out_dir, "sample_level.npy"), sample_out)
    weights_path = os.path.join(out_dir, "gan_final.weights.h5")
    G.save_weights(weights_path)
    print("Échantillon et poids sauvegardés dans", out_dir)

    # 8) Export SavedModel pour Java
    saved_model_dir = os.path.join(os.path.dirname(__file__), "gan_savedmodel")
    tf.keras.models.save_model(
        G,
        saved_model_dir,
        overwrite=True,
        include_optimizer=False,
        save_format="tf"
    )
    print("SavedModel exporté dans", saved_model_dir)


if __name__ == "__main__":
    args = parse_args()
    train(
        args.epochs,
        args.batch_size,
        args.latent_dim,
        args.orig_glob,
        args.width,
        args.out_dir
    )
