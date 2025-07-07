#!/usr/bin/env python3
# models/gan_lsi/train_keras.py

import os
import argparse
import numpy as np
import tensorflow as tf
from .gan.model_keras import build_generator, build_discriminator  # import relatif

def parse_args():
    p = argparse.ArgumentParser(description="Mini‐entraînement Keras pour MarioGAN")
    p.add_argument("--epochs",     type=int, default=3,  help="Nombre d’epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Taille de batch")
    p.add_argument("--latent-dim", type=int, default=32, help="Dimension du vecteur latent")
    p.add_argument("--out-dir",    type=str, default=os.path.dirname(__file__),
                   help="Dossier de sortie (sample + checkpoints)")
    return p.parse_args()

def train(epochs, batch_size, latent_dim, out_dir):
    # 1) Prépare les dossiers
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2) Construit G et D
    G = build_generator(latent_dim=latent_dim)
    D = build_discriminator()
    D.compile(optimizer=tf.keras.optimizers.RMSprop(5e-5),
              loss="binary_crossentropy")

    # 3) Monte le GAN combiné
    z = tf.keras.Input(shape=(latent_dim,))
    gan_out = D(G(z))
    GAN = tf.keras.Model(z, gan_out)
    GAN.compile(optimizer=tf.keras.optimizers.RMSprop(5e-5),
                loss="binary_crossentropy")

    # 4) Dataset factice
    real_shape = G.output_shape[1:]  # e.g. (16,256,1)
    real_data  = np.random.randn(100, *real_shape).astype("float32")
    dataset    = tf.data.Dataset.from_tensor_slices(real_data)
    dataset    = dataset.shuffle(100).batch(batch_size)

    # 5) Entraînement
    for epoch in range(1, epochs+1):
        for real_batch in dataset:
            # 5a) Entraîne D sur vrais
            D.train_on_batch(real_batch, np.ones((real_batch.shape[0],1)))
            # 5b) Entraîne D sur fake
            z_sample   = np.random.randn(real_batch.shape[0], latent_dim).astype("float32")
            fake_batch = G.predict(z_sample, verbose=0)
            D.train_on_batch(fake_batch, np.zeros((real_batch.shape[0],1)))
            # 5c) Entraîne G pour tromper D
            z2 = np.random.randn(batch_size, latent_dim).astype("float32")
            GAN.train_on_batch(z2, np.ones((batch_size,1)))
        print(f"Epoch {epoch}/{epochs} terminé")

    # 6) Sauvegarde un échantillon
    sample = G.predict(np.random.randn(1, latent_dim).astype("float32"))
    sample_path = os.path.join(out_dir, "sample_level.npy")
    np.save(sample_path, sample)
    print("Échantillon sauvegardé dans", sample_path)

    # 7) Sauvegarde les poids (HDF5, extension .weights.h5)
    weights_path = os.path.join(ckpt_dir, "gan_final.weights.h5")
    G.save_weights(weights_path)  # Keras déduit le format de l’extension
    print("Poids du générateur sauvegardés dans", weights_path)

if __name__ == "__main__":
    args = parse_args()
    train(args.epochs, args.batch_size, args.latent_dim, args.out_dir)
