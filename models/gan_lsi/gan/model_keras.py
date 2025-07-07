#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_generator(latent_dim=32,
                    out_shape=(19, 256),
                    base_filters=64,
                    n_symbols=32):
    h, w = out_shape
    model = Sequential(name="Generator")
    model.add(layers.Input(shape=(latent_dim,)))

    model.add(layers.Dense(4 * 4 * base_filters * 8, use_bias=False))
    model.add(layers.Reshape((4, 4, base_filters * 8)))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(base_filters * 4, 4, 2, "same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(base_filters * 2, 4, 2, "same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())

    model.add(layers.Lambda(lambda x: tf.image.resize(x, [h, w])))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())

    model.add(layers.Conv2D(n_symbols, 1, 1, "same", use_bias=False))
    return model

def build_discriminator(base_filters=64):
    """
    Discriminateur sans couche Input : 
    la forme est inférée au premier appel.
    """
    model = Sequential(name="Discriminator")
    model.add(layers.Conv2D(base_filters, (4, 16), (1, 16), "valid"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(base_filters * 2, 4, 2, "same"))
    model.add(layers.BatchNormalization()); model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
