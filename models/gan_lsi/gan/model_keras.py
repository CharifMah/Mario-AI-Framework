import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim=32, out_shape=(16,256,1), base_filters=64):
    model = tf.keras.Sequential(name="Generator")
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(4*4*base_filters*8, use_bias=False))
    model.add(layers.Reshape((4, 4, base_filters*8)))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(base_filters*4, 4, 2, "same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(base_filters*2, 4, 2, "same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    # ==> corrigé ici :
    model.add(layers.Conv2DTranspose(
        base_filters,
        kernel_size=(1,16),
        strides=(1,16),
        padding="valid",
        use_bias=False
    ))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(out_shape[-1], 3, 1, "same", activation="tanh"))
    return model

def build_discriminator(in_shape=(16,256,1), base_filters=64):
    model = tf.keras.Sequential(name="Discriminator")
    model.add(layers.Input(shape=in_shape))
    model.add(layers.Conv2D(base_filters, (4,16), (1,16), "valid"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(base_filters*2, 4, 2, "same"))
    model.add(layers.BatchNormalization()); model.add(layers.LeakyReLU(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
