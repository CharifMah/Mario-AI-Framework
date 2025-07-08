#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gain_train_balanced_v5.py

Même GAN, en forçant un dtype unique dès l’entrée du train_step
pour éviter tout mélange float16/float32.
"""
import os
import tensorflow as tf

from data_utils import (
    load_levels,
    build_vocabulary,
    encode_levels,
    save_mapping,
    HEIGHT,
    WIDTH
)

# 1) TF & GPU
print("TensorFlow version :", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("→ GPU détecté :", gpus)
else:
    print("→ Aucun GPU, CPU uniquement")

# 2) Mixed precision
from tensorflow.keras import mixed_precision
use_fp16 = bool(gpus)
policy = mixed_precision.Policy('mixed_float16' if use_fp16 else 'float32')
mixed_precision.set_global_policy(policy)
print("→ Politique de précision :", policy)

# 3) Strategy
strategy = (tf.distribute.MirroredStrategy() if len(gpus) > 1
            else tf.distribute.get_strategy())
if isinstance(strategy, tf.distribute.MirroredStrategy):
    print("→ MirroredStrategy activée")

# 4) Hyperparamètres
LATENT_DIM      = 32
BATCH_SIZE      = 64 if use_fp16 else 32
EPOCHS          = 1000
LR_D            = 1e-5
LR_G            = 3e-4
LABEL_SMOOTHING = 0.3
NOISE_STDDEV    = 0.07
GEN_UPDATES     = 4

# 5) Chargement données
levels = load_levels(os.path.join("..","..","levels","hopper"))
print(f"{len(levels)} niveaux chargés")
vocab, c2i, i2c = build_vocabulary(levels)
X = encode_levels(levels, c2i)
X_onehot = tf.one_hot(X, len(vocab))
print("One-hot shape:", X_onehot.shape)

# 6) Modèles & optims
with strategy.scope():
    def make_generator():
        m = tf.keras.Sequential([
            tf.keras.layers.Input(LATENT_DIM),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(HEIGHT*WIDTH*len(vocab)),
            tf.keras.layers.Reshape((HEIGHT, WIDTH, len(vocab))),
            tf.keras.layers.Softmax(axis=-1),
        ], name="Generator")
        return m

    def make_discriminator():
        m = tf.keras.Sequential([
            tf.keras.layers.Input((HEIGHT, WIDTH, len(vocab))),
            tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(1,activation='sigmoid'),
        ], name="Discriminator")
        return m

    G = make_generator()
    D = make_discriminator()

    # optimizers
    if use_fp16:
        base_d = tf.keras.optimizers.Adam(LR_D)
        base_g = tf.keras.optimizers.Adam(LR_G)
        d_opt = mixed_precision.LossScaleOptimizer(base_d)
        g_opt = mixed_precision.LossScaleOptimizer(base_g)
    else:
        d_opt = tf.keras.optimizers.Adam(LR_D)
        g_opt = tf.keras.optimizers.Adam(LR_G)

    bce = tf.keras.losses.BinaryCrossentropy()

# 7) train_step
@tf.function(experimental_compile=True)
def train_step(real_batch):
    # On force tout en compute_dtype (float16 si mixed)
    compute_dtype = policy.compute_dtype
    real = tf.cast(real_batch, compute_dtype)

    bs = tf.shape(real)[0]
    # bruits & labels en compute_dtype
    noise_d = tf.random.normal([bs, LATENT_DIM], dtype=compute_dtype)
    real_noise = tf.random.normal(tf.shape(real),
                                  stddev=NOISE_STDDEV,
                                  dtype=compute_dtype)

    # --- Update D ---
    with tf.GradientTape() as td:
        fake = G(noise_d, training=True)
        fake_noise = tf.random.normal(tf.shape(fake),
                                      stddev=NOISE_STDDEV,
                                      dtype=compute_dtype)

        real_noisy = real + real_noise
        fake_noisy = fake + fake_noise

        real_logits = D(real_noisy, training=True)
        fake_logits = D(fake_noisy, training=True)

        real_labels = tf.random.uniform(tf.shape(real_logits),
                                        minval=1.-LABEL_SMOOTHING,
                                        maxval=1.,
                                        dtype=compute_dtype)
        fake_labels = tf.random.uniform(tf.shape(fake_logits),
                                        minval=0.,
                                        maxval=LABEL_SMOOTHING,
                                        dtype=compute_dtype)

        d_loss = bce(real_labels, real_logits) + bce(fake_labels, fake_logits)
        d_scaled = d_opt.get_scaled_loss(d_loss) if use_fp16 else d_loss

    grads_d = td.gradient(d_scaled, D.trainable_variables)
    if use_fp16:
        grads_d = d_opt.get_unscaled_gradients(grads_d)
    d_opt.apply_gradients(zip(grads_d, D.trainable_variables))

    # --- Update G x GEN_UPDATES ---
    total_g = tf.constant(0., dtype=compute_dtype)
    for _ in range(GEN_UPDATES):
        noise_g = tf.random.normal([bs, LATENT_DIM], dtype=compute_dtype)
        with tf.GradientTape() as tg:
            fake2 = G(noise_g, training=True)
            logits2 = D(fake2, training=True)
            # on veut 1.0 labels
            ones = tf.ones_like(logits2, dtype=compute_dtype)
            g_loss = bce(ones, logits2)
            total_g += g_loss
            g_scaled = g_opt.get_scaled_loss(g_loss) if use_fp16 else g_loss

        grads_g = tg.gradient(g_scaled, G.trainable_variables)
        if use_fp16:
            grads_g = g_opt.get_unscaled_gradients(grads_g)
        g_opt.apply_gradients(zip(grads_g, G.trainable_variables))

    return d_loss, total_g

# 8) Pipeline tf.data
ds = (tf.data.Dataset.from_tensor_slices(X_onehot)
      .shuffle(1_000)
      .batch(BATCH_SIZE, drop_remainder=True)
      .cache()
      .repeat()
      .prefetch(tf.data.AUTOTUNE))
steps = len(X) // BATCH_SIZE
it = iter(ds)

# 9) Warm-up
print("🔄 Warm-up…")
_ = train_step(next(it))
print("✅ Warm-up OK")

# 10) Training loop
for ep in range(1, EPOCHS+1):
    for st in range(1, steps+1):
        d_l, g_l = train_step(next(it))
        print(f"Epoch {ep}/{EPOCHS} | step {st}/{steps} "
              f"| d_loss={d_l:.4f} | g_loss={g_l:.4f}")
    print(f"→ Fin époque {ep}")

# 11) Save
print("Sauvegarde…")
G.save("gan_savedmodel")
save_mapping(c2i, i2c, "char_mapping.json")
print("Terminé.")
