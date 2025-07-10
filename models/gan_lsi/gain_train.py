#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gain_train_best.py

WGAN-GP : architecture riche + ExponentialDecay + mixed precision + SpectralNorm
Hyperparamètres optimaux validés :
 - LATENT_DIM   = 64
 - BATCH_SIZE   = 32
 - EPOCHS       = 100
 - LR_C_INIT    = 1e-5
 - LR_G_INIT    = 5e-5
 - DECAY_RATE   = 0.95
 - LAMBDA_GP    = 6.0
 - N_CRITIC     = 2
"""
import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from data_utils import load_levels, build_vocabulary, encode_levels, save_mapping, HEIGHT, WIDTH
try:
    from tensorflow_addons.layers import SpectralNormalization
except ImportError:
    SpectralNormalization = lambda layer: layer

# --- Logging setup ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir  = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"train_{ts}.log")

logger = logging.getLogger("WGAN_Best")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s", datefmt="%H:%M:%S")
ch = logging.StreamHandler();    ch.setFormatter(fmt)
fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)

# --- Hyperparameters -------------------------------------------------------
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS     = 100

LR_C_INIT  = 1e-5         # critic learning rate
LR_G_INIT  = 5e-5         # generator learning rate
DECAY_RATE = 0.95
LAMBDA_GP  = 6.0          # gradient penalty coefficient
N_CRITIC   = 2            # critic updates per generator update

logger.info("---- Hyperparameters ----")
logger.info("LATENT_DIM   = %d", LATENT_DIM)
logger.info("BATCH_SIZE   = %d", BATCH_SIZE)
logger.info("EPOCHS       = %d", EPOCHS)
logger.info("LR_C_INIT    = %.1e", LR_C_INIT)
logger.info("LR_G_INIT    = %.1e", LR_G_INIT)
logger.info("DECAY_RATE   = %.3f", DECAY_RATE)
logger.info("LAMBDA_GP    = %.1f", LAMBDA_GP)
logger.info("N_CRITIC     = %d", N_CRITIC)
logger.info("-------------------------")

# --- GPU & precision ------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
policy = 'mixed_float16' if gpus else 'float32'
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy)
strategy = tf.distribute.MirroredStrategy() if len(gpus)>1 else tf.distribute.get_strategy()
logger.info("TF %s, Policy %s, Strategy %s", tf.__version__, policy, type(strategy).__name__)

# --- Data loading ----------------------------------------------------------
levels = load_levels(os.path.join("..","..","levels","hopper"))
vocab, c2i, i2c = build_vocabulary(levels)
vocab_size = len(vocab)
logger.info("%d niveaux chargés, vocab_size=%d", len(levels), vocab_size)

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
logger.info("Steps per epoch: %d", steps_per_epoch)

# --- Learning rate schedules ----------------------------------------------
lr_schedule_C = ExponentialDecay(LR_C_INIT, steps_per_epoch, DECAY_RATE, staircase=True)
lr_schedule_G = ExponentialDecay(LR_G_INIT, steps_per_epoch, DECAY_RATE, staircase=True)

# --- Model definitions -----------------------------------------------------
with strategy.scope():
    def make_generator():
        inp = layers.Input((LATENT_DIM,))
        x   = layers.Dense(512, activation='relu')(inp)
        x   = layers.LayerNormalization()(x)
        x   = layers.Dense(HEIGHT*WIDTH*vocab_size)(x)
        x   = layers.Reshape((HEIGHT, WIDTH, vocab_size))(x)
        out = layers.Softmax(axis=-1)(x)
        return tf.keras.Model(inp, out, name="Generator")

    def make_critic():
        inp = layers.Input((HEIGHT, WIDTH, vocab_size))
        x = SpectralNormalization(layers.Conv2D(64, 3, padding='same', activation='relu'))(inp)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = SpectralNormalization(layers.Conv2D(128, 3, padding='same', activation='relu'))(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inp, out, name="Critic")

    G = make_generator()
    C = make_critic()
    optC = tf.keras.optimizers.Adam(lr_schedule_C, beta_1=0.5, beta_2=0.9)
    optG = tf.keras.optimizers.Adam(lr_schedule_G, beta_1=0.5, beta_2=0.9)

    # Metrics
    m_real = tf.keras.metrics.Mean('C_real')
    m_fake = tf.keras.metrics.Mean('C_fake')
    m_gp   = tf.keras.metrics.Mean('GP')
    m_c    = tf.keras.metrics.Mean('LossC')
    m_g    = tf.keras.metrics.Mean('LossG')

    # Gradient penalty
    def gradient_penalty(real, fake):
        real_f = tf.cast(real, tf.float32)
        fake_f = tf.cast(fake, tf.float32)
        alpha  = tf.random.uniform([tf.shape(real_f)[0],1,1,1], dtype=tf.float32)
        interp = real_f + alpha * (fake_f - real_f)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = C(interp, training=True)
        grads = tape.gradient(pred, interp)
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]))
        return tf.reduce_mean((norm - 1.0) ** 2)

# --- Training step ---------------------------------------------------------
@tf.function
def train_step(batch):
    for _ in range(N_CRITIC):
        z = tf.random.normal((BATCH_SIZE, LATENT_DIM))
        with tf.GradientTape() as tC:
            fake = G(z, training=True)
            cr   = tf.cast(C(batch, training=True), tf.float32)
            cf   = tf.cast(C(fake, training=True), tf.float32)
            gp   = gradient_penalty(batch, fake)
            lossC = tf.reduce_mean(cf) - tf.reduce_mean(cr) + LAMBDA_GP * gp
        gradsC = tC.gradient(lossC, C.trainable_variables)
        optC.apply_gradients(zip(gradsC, C.trainable_variables))

    z = tf.random.normal((BATCH_SIZE, LATENT_DIM))
    with tf.GradientTape() as tG:
        fake  = G(z, training=True)
        gf    = tf.cast(C(fake, training=True), tf.float32)
        lossG = -tf.reduce_mean(gf)
    gradsG = tG.gradient(lossG, G.trainable_variables)
    optG.apply_gradients(zip(gradsG, G.trainable_variables))

    m_real.update_state(cr)
    m_fake.update_state(cf)
    m_gp.update_state(gp)
    m_c.update_state(lossC)
    m_g.update_state(lossG)

# --- Training loop ---------------------------------------------------------
logger.info("-> Warm-up compilation…")
_ = train_step(next(iter(dataset)))
logger.info("-> Start training…")
for epoch in range(1, EPOCHS+1):
    m_real.reset_states(); m_fake.reset_states()
    m_gp.reset_states(); m_c.reset_states(); m_g.reset_states()
    for _ in range(steps_per_epoch):
        train_step(next(iter(dataset)))
    logger.info(
        "Epoch %d/%d — C_real=%.4f — C_fake=%.4f — GP=%.4f — LossC=%.4f — LossG=%.4f",
        epoch, EPOCHS,
        m_real.result(), m_fake.result(),
        m_gp.result(), m_c.result(), m_g.result()
    )

# --- Save -----------------------------------------------------------------
ckpt_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
G.save(os.path.join(ckpt_dir, "gan_savedmodel"))
save_mapping(c2i, i2c, "char_mapping.json")
logger.info("✔️ Model and mapping saved.")
