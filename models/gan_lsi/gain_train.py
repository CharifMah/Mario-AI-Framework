#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gain_train.py

Wasserstein GAN with Gradient Penalty pour niveaux Mario :
 - Critic linéaire (pas de sigmoïde)
 - Loss Wasserstein : E[C(fake)]−E[C(real)] + λ·GP
 - Gradient penalty (λ=10)
 - N_CRITIC steps par update du générateur
 - Learning rates séparés (C:1e-5, G:3e-4)
 - Mixed precision si GPU
 - MirroredStrategy multi-GPU
 - XLA compilation du train_step
 - Logging complet par batch en ASCII
"""
import os
import tensorflow as tf
import numpy as np
from data_utils import load_levels, build_vocabulary, encode_levels, save_mapping, HEIGHT, WIDTH

# 1) GPU & mixed precision
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("-> GPU détecté:", gpus)
policy = 'mixed_float16' if gpus else 'float32'
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy)
print("-> DType policy:", policy)
strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

# 2) Hyper-paramètres
LATENT_DIM   = 32
BATCH_SIZE   = 64 if gpus else 32
EPOCHS       = 500
LR_C         = 1e-5
LR_G         = 3e-4
N_CRITIC     = 5     # updates du critic par update du générateur
LAMBDA_GP    = 10.0  # coefficient du gradient penalty

# 3) Data
levels = load_levels(os.path.join("..","..","levels","hopper"))
vocab, c2i, i2c = build_vocabulary(levels)
vocab_size = len(vocab)
print(f"{len(levels)} niveaux chargés, vocab size = {vocab_size}")
X = encode_levels(levels, c2i)
X = tf.one_hot(X, vocab_size)  # (N, H, W, V)
dataset = (
    tf.data.Dataset.from_tensor_slices(X)
      .shuffle(1000)
      .batch(BATCH_SIZE, drop_remainder=True)
      .repeat()
      .prefetch(tf.data.AUTOTUNE)
)
steps_per_epoch = len(levels) // BATCH_SIZE
it = iter(dataset)

with strategy.scope():
    # 4) Modèles
    def make_generator():
        inp = tf.keras.Input((LATENT_DIM,))
        x = tf.keras.layers.Dense(256, activation='relu')(inp)
        x = tf.keras.layers.Dense(HEIGHT * WIDTH * vocab_size)(x)
        x = tf.keras.layers.Reshape((HEIGHT, WIDTH, vocab_size))(x)
        out = tf.keras.layers.Softmax(axis=-1)(x)
        return tf.keras.Model(inp, out, name="Generator")

    def make_critic():
        inp = tf.keras.Input((HEIGHT, WIDTH, vocab_size))
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        out = tf.keras.layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inp, out, name="Critic")

    G = make_generator()
    C = make_critic()
    optC = tf.keras.optimizers.Adam(LR_C, beta_1=0.5, beta_2=0.9)
    optG = tf.keras.optimizers.Adam(LR_G, beta_1=0.5, beta_2=0.9)

    # fonction pour le gradient penalty (calcul en float32)
    def gradient_penalty(real, fake):
        real_f = tf.cast(real, tf.float32)
        fake_f = tf.cast(fake, tf.float32)
        alpha = tf.random.uniform([tf.shape(real_f)[0], 1, 1, 1],
                                   0.0, 1.0, dtype=real_f.dtype)
        interp = real_f + alpha * (fake_f - real_f)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interp)
            pred = C(interp, training=True)
        grads = gp_tape.gradient(pred, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))
        gp = tf.reduce_mean((norm - 1.0)**2)
        return gp

# 5) train_step compilé XLA
@tf.function(experimental_compile=True)
def train_step(real_batch):
    # --- Critic updates ---
    for _ in range(N_CRITIC):
        noise = tf.random.normal((BATCH_SIZE, LATENT_DIM), dtype=real_batch.dtype)
        with tf.GradientTape() as tapeC:
            fake = G(noise, training=True)
            real_score = C(real_batch, training=True)
            fake_score = C(fake, training=True)
            gp = gradient_penalty(real_batch, fake)

            # on remet tout en float16 pour la lossC
            gp16 = tf.cast(gp, real_score.dtype)
            lambda16 = tf.cast(LAMBDA_GP, real_score.dtype)
            lossC = (tf.reduce_mean(fake_score)
                     - tf.reduce_mean(real_score)
                     + lambda16 * gp16)

        gradsC = tapeC.gradient(lossC, C.trainable_variables)
        optC.apply_gradients(zip(gradsC, C.trainable_variables))

    # --- Generator update ---
    noise = tf.random.normal((BATCH_SIZE, LATENT_DIM), dtype=real_batch.dtype)
    with tf.GradientTape() as tapeG:
        fake = G(noise, training=True)
        fake_score = C(fake, training=True)
        lossG = -tf.reduce_mean(fake_score)
    gradsG = tapeG.gradient(lossG, G.trainable_variables)
    optG.apply_gradients(zip(gradsG, G.trainable_variables))

    return lossC, lossG

# 6) Warm-up
print("🔄 Warm-up compilation…")
_ = train_step(next(it))
print("✅ Warm-up terminé, début entraînement.")

# 7) Boucle d’entraînement
for epoch in range(1, EPOCHS+1):
    for step in range(1, steps_per_epoch+1):
        c_l, g_l = train_step(next(it))
        print(f"Epoch {epoch}/{EPOCHS} - step {step}/{steps_per_epoch} - c={c_l:.4f} - g={g_l:.4f}")
    print(f"→ Epoch {epoch} terminé: last c={c_l:.4f}, g={g_l:.4f}")

# 8) Sauvegarde
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
G.save("gan_savedmodel")
save_mapping(c2i, i2c, "char_mapping.json")
print("✔️ Modèle et mapping sauvegardés.")

