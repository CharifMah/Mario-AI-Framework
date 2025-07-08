#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gan_tf_basic_improved_all_steps.py

Même GAN « classique » pour niveaux Mario, avec :
 - Label smoothing (±30 %) et bruit (σ=0.07) sur les entrées du discriminateur
 - 4 updates du générateur par update du discriminateur
 - Learning rates séparés (D:1e-5, G:3e-4)
 - Mixed precision si GPU
 - MirroredStrategy multi-GPU
 - XLA compilation du train_step
 - tf.data pipeline optimisée
 - Logging à chaque step
 - Checkpoints en SavedModel format
"""
import os
import tensorflow as tf
from datetime import datetime
from data_utils import load_levels, build_vocabulary, encode_levels, save_mapping, HEIGHT, WIDTH

# 1) GPU & mixed precision
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("→ GPU détecté :", gpus)
policy = 'mixed_float16' if gpus else 'float32'
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy)
print("→ DType policy :", policy)
strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

# 2) Hyper-paramètres
LATENT_DIM      = 32
BATCH_SIZE      = 64 if gpus else 32
EPOCHS          = 500
LR_D            = 1e-5
LR_G            = 3e-4
LABEL_SMOOTHING = 0.3
NOISE_STDDEV    = 0.07
GEN_UPDATES     = 4
CKPT_FREQ       = 100       # checkpoint toutes les CKPT_FREQ époques

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

# 4) Models & optimizers
with strategy.scope():
    def make_generator():
        inp = tf.keras.Input((LATENT_DIM,))
        x = tf.keras.layers.Dense(256, activation='relu')(inp)
        x = tf.keras.layers.Dense(HEIGHT * WIDTH * vocab_size)(x)
        x = tf.keras.layers.Reshape((HEIGHT, WIDTH, vocab_size))(x)
        return tf.keras.Model(inp, tf.keras.layers.Softmax(axis=-1)(x), name="Generator")

    def make_discriminator():
        inp = tf.keras.Input((HEIGHT, WIDTH, vocab_size))
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        return tf.keras.Model(inp, tf.keras.layers.Dense(1, activation='sigmoid')(x), name="Discriminator")

    G   = make_generator()
    D   = make_discriminator()
    optD = tf.keras.optimizers.Adam(LR_D)
    optG = tf.keras.optimizers.Adam(LR_G)
    bce  = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# 5) TensorBoard & Checkpointing
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.create_file_writer(logdir)
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# 6) train_step compilé XLA
@tf.function(experimental_compile=True)
def train_step(real_batch):
    bs = tf.shape(real_batch)[0]

    # — Discriminator —
    noise = tf.random.normal((bs, LATENT_DIM), dtype=real_batch.dtype)
    with tf.GradientTape() as td:
        fake       = G(noise, training=True)
        real_noisy = real_batch + tf.random.normal(
            tf.shape(real_batch), stddev=NOISE_STDDEV, dtype=real_batch.dtype)
        fake_noisy = fake + tf.random.normal(
            tf.shape(fake), stddev=NOISE_STDDEV, dtype=fake.dtype)
        real_lbl = tf.random.uniform((bs,1), 1.-LABEL_SMOOTHING, 1.0,
                                     dtype=real_noisy.dtype)
        fake_lbl = tf.random.uniform((bs,1), 0.0, LABEL_SMOOTHING,
                                     dtype=fake_noisy.dtype)
        real_out = D(real_noisy, training=True)
        fake_out = D(fake_noisy, training=True)
        lossD    = bce(real_lbl, real_out) + bce(fake_lbl, fake_out)
    gradsD = td.gradient(lossD, D.trainable_variables)
    optD.apply_gradients(zip(gradsD, D.trainable_variables))

    # — Generator (x GEN_UPDATES) —
    lossG = tf.constant(0.0, dtype=tf.float32)
    for _ in range(GEN_UPDATES):
        noise = tf.random.normal((bs, LATENT_DIM), dtype=real_batch.dtype)
        with tf.GradientTape() as tg:
            out    = D(G(noise, training=True), training=True)
            loss16 = bce(tf.ones_like(out), out)
        gradsG = tg.gradient(loss16, G.trainable_variables)
        optG.apply_gradients(zip(gradsG, G.trainable_variables))
        lossG += tf.cast(loss16, tf.float32)

    return lossD, lossG

# 7) Warm-up
print("🔄 Warm-up compilation…")
it = iter(dataset)
_ = train_step(next(it))
print("✅ Warm-up terminé — démarrage entraînement.")

# 8) Boucle d’entraînement
global_step = 0
for epoch in range(1, EPOCHS+1):
    for step in range(1, steps_per_epoch+1):
        global_step += 1
        real_batch = next(it)
        lossD, lossG = train_step(real_batch)

        # log TensorBoard + console à chaque step
        with writer.as_default():
            tf.summary.scalar("lossD", lossD, step=global_step)
            tf.summary.scalar("lossG", lossG, step=global_step)
        print(f"Epoch {epoch}/{EPOCHS} · step {step}/{steps_per_epoch} "
              f"(global {global_step}) · d={lossD:.4f} · g={lossG:.4f}")

    # checkpoint en SavedModel (identique à votre ancien G.save)
    if epoch % CKPT_FREQ == 0 or epoch == EPOCHS:
        ckpt_path = os.path.join(ckpt_dir, f"G_epoch_{epoch}")
        G.save(ckpt_path)
        print(f"→ Checkpoint ép {epoch} sauvegardé dans {ckpt_path}")

# 9) Sauvegarde finale
G.save("gan_savedmodel")              # crée le dossier gan_savedmodel/
save_mapping(c2i, i2c, "char_mapping.json")
print("✔️ Entraînement terminé. Modèle et mapping sauvegardés.")
