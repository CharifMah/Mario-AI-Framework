#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gain_train.py

Wasserstein GAN with Gradient Penalty pour niveaux Mario,
avec logging structuré, gestion des erreurs CuDNN,
limitation GPU (memory_limit + memory_growth safe).
"""
import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.python.framework.errors_impl import UnknownError
from data_utils import (
    load_levels, build_vocabulary, encode_levels,
    save_mapping, HEIGHT, WIDTH
)

# --- Configuration du logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"wgan-gp_train_{now}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. File: {log_file}")

# --- 1) GPU config: first memory_limit, then safe memory_growth ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 1a) Virtual device avec limite fixe
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        logger.info("GPU: memory_limit=4096 MB on GPU:0")
    except RuntimeError as e:
        logger.warning(f"Could not set memory_limit: {e}")

    # 1b) Puis tenter memory_growth
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
            logger.info(f"GPU: memory_growth enabled on {g.name}")
        except ValueError:
            logger.warning(f"Skipping memory_growth on {g.name} (virtual device configured)")

# --- Mixed precision policy ---
policy = 'mixed_float16' if gpus else 'float32'
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy)
logger.info(f"Precision policy: {policy}")

# --- Distribution strategy ---
strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
logger.info(f"Using strategy: {strategy.__class__.__name__}")

# --- 2) Hyper-paramètres  ---
LATENT_DIM = 64
BATCH_SIZE = 64 if gpus else 32
EPOCHS     = 500

# Critic
LR_C      = 5e-6
N_CRITIC  = 7
LAMBDA_GP = 20.0

# Generator
LR_G      = 3e-4




logger.info(
    f"Hyperparams: LATENT_DIM={LATENT_DIM}, BATCH_SIZE={BATCH_SIZE},"
    f" EPOCHS={EPOCHS}, LR_C={LR_C}, LR_G={LR_G}, N_CRITIC={N_CRITIC},"
    f" LAMBDA_GP={LAMBDA_GP}"
)

# --- 3) Préparation des données ---
levels = load_levels(os.path.join("..","..","levels","hopper"))
vocab, c2i, i2c = build_vocabulary(levels)
vocab_size = len(vocab)
logger.info(f"{len(levels)} levels loaded, vocab_size={vocab_size}")
X = encode_levels(levels, c2i)
X = tf.one_hot(X, vocab_size)

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
    # --- Modèles ---
    def make_generator():
        inp = tf.keras.Input((LATENT_DIM,))
        x = tf.keras.layers.Dense(256, activation='relu')(inp)
        x = tf.keras.layers.Dense(HEIGHT * WIDTH * vocab_size)(x)
        x = tf.keras.layers.Reshape((HEIGHT, WIDTH, vocab_size))(x)
        return tf.keras.Model(inp, x, name="Generator") 

    def make_critic():
        inp = tf.keras.Input((HEIGHT, WIDTH, vocab_size))
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        return tf.keras.Model(inp, tf.keras.layers.Dense(1, activation='linear')(x), name="Critic")

    G, C = make_generator(), make_critic()
    logger.info("Generator architecture:"); G.summary(print_fn=lambda s: logger.info(s))
    logger.info("Critic architecture:");    C.summary(print_fn=lambda s: logger.info(s))

    base_optC = tf.keras.optimizers.Adam(LR_C, beta_1=0.5, beta_2=0.9)
    base_optG = tf.keras.optimizers.Adam(LR_G, beta_1=0.5, beta_2=0.9)
    optC = mixed_precision.LossScaleOptimizer(base_optC)
    optG = mixed_precision.LossScaleOptimizer(base_optG)

    def gradient_penalty(real, fake):
        real_f = tf.cast(real, tf.float32)
        fake_f = tf.cast(fake, tf.float32)
        alpha = tf.random.uniform([tf.shape(real_f)[0], 1, 1, 1], dtype=real_f.dtype)
        interp = real_f + alpha * (fake_f - real_f)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interp)
            pred = C(interp, training=True)
        grads = gp_tape.gradient(pred, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))
        return tf.reduce_mean((norm - 1.0)**2)

    @tf.function(experimental_compile=False)
    def train_step(real_batch):
        # Critic
        for _ in range(N_CRITIC):
            noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))
            with tf.GradientTape() as tapeC:
                fake = G(noise, training=True)
                r_score = C(real_batch, training=True)
                f_score = C(fake, training=True)
                gp = gradient_penalty(real_batch, fake)
                gp_c = tf.cast(gp, f_score.dtype)
                lossC = tf.reduce_mean(f_score) - tf.reduce_mean(r_score) + tf.cast(LAMBDA_GP, f_score.dtype) * gp_c
                scaled_lossC = optC.get_scaled_loss(lossC)
            gradsC = tapeC.gradient(scaled_lossC, C.trainable_variables)
            gradsC = optC.get_unscaled_gradients(gradsC)
            gradsC = [tf.clip_by_norm(g, 1.0) for g in gradsC]
            optC.apply_gradients(zip(gradsC, C.trainable_variables))

        # Generator
        noise = tf.random.normal((BATCH_SIZE, LATENT_DIM))
        with tf.GradientTape() as tapeG:
            fake = G(noise, training=True)
            lossG = -tf.reduce_mean(C(fake, training=True))
            scaled_lossG = optG.get_scaled_loss(lossG)
        gradsG = tapeG.gradient(scaled_lossG, G.trainable_variables)
        gradsG = optG.get_unscaled_gradients(gradsG)
        gradsG = [tf.clip_by_norm(g, 1.0) for g in gradsG]
        optG.apply_gradients(zip(gradsG, G.trainable_variables))

        return tf.reduce_mean(r_score), tf.reduce_mean(f_score), gp, lossC, lossG

# Warm-up
logger.info("Warm-up compilation…")
_ = train_step(next(it))
logger.info("Warm-up terminé, début entraînement.")

# Training loop
try:
    for epoch in range(1, EPOCHS + 1):
        sum_r = sum_f = sum_gp = sum_c = sum_g = 0.0
        for _ in range(steps_per_epoch):
            r, f, gp, lc, lg = train_step(next(it))
            sum_r += r; sum_f += f; sum_gp += gp; sum_c += lc; sum_g += lg
        logger.info(
            f"Epoch {epoch}/{EPOCHS} — "
            f"C_real={sum_r/steps_per_epoch:.4f} — "
            f"C_fake={sum_f/steps_per_epoch:.4f} — "
            f"GP={sum_gp/steps_per_epoch:.4f} — "
            f"LossC={sum_c/steps_per_epoch:.4f} — "
            f"LossG={sum_g/steps_per_epoch:.4f}"
        )
except UnknownError as e:
    logger.error("=== Interrupted by a CuDNN internal error ===")
    logger.exception(e)
    interrupted_dir = f"gan_savedmodel_interrupted_{now}"
    os.makedirs(interrupted_dir, exist_ok=True)
    G.save(interrupted_dir)
    save_mapping(c2i, i2c, f"char_mapping_interrupted_{now}.json")
    logger.info(f"Model and mapping saved to {interrupted_dir}")
    exit(0)

# Final save
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
G.save("gan_savedmodel")
save_mapping(c2i, i2c, "char_mapping.json")
logger.info("Modèle et mapping sauvegardés.")
