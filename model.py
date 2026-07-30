"""
Convolutional Variational Autoencoder (CVAE) for magpie spectrograms.
====================================================================
Defines the encoder/decoder/VAE classes. Imported by train.py.

A VAE (rather than a vanilla autoencoder) is used because our pipeline averages
latent vectors across a cohort -- urban vs rural -- and decodes the mean to get
a synthesised "average call" for spectral-peak analysis. A VAE forces the
latent space to be SMOOTH and CONTINUOUS (via the KL divergence loss), which
means averaging in it produces a coherent decoded spectrogram instead of the
blurry mush a vanilla AE can give.

Input shape:  (256, 256, 1)  -- Log-Mel spectrograms from preprocess.py.
Latent size:  128 dims (configurable).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --- ARCHITECTURE CONFIG ---
INPUT_SHAPE = (256, 256, 1)   # (mel bands, time frames, channel)
LATENT_DIM = 32               # ~108 samples -- keep this small to avoid overfit
BASE_FILTERS = 16             # halved from 32; further reduces parameter count
DROPOUT_RATE = 0.35           # raised from 0.25 for stronger regularisation


# --- SAMPLING LAYER (the "reparameterisation trick") ---
class Sampling(layers.Layer):
    """Draws z = mu + sigma * epsilon, so gradients can flow through sampling.
    This is what makes a VAE differentiable end-to-end."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder():
    """Spectrogram -> (z_mean, z_log_var, z). Four Conv2D blocks downsample
    256x256 -> 16x16, then flatten and project to the latent space."""
    inputs = keras.Input(shape=INPUT_SHAPE, name="spectrogram")
    x = inputs
    filters = BASE_FILTERS
    for i in range(4):  # 256 -> 128 -> 64 -> 32 -> 16
        x = layers.Conv2D(filters, 3, strides=2, padding="same",
                          name=f"enc_conv_{i}")(x)
        x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
        x = layers.LeakyReLU(0.2, name=f"enc_act_{i}")(x)
        x = layers.Dropout(DROPOUT_RATE, name=f"enc_drop_{i}")(x)
        filters *= 2
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu", name="enc_dense")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder():
    """Latent vector -> spectrogram. Mirrors the encoder with Conv2DTranspose."""
    latent_inputs = keras.Input(shape=(LATENT_DIM,), name="z_in")
    x = layers.Dense(16 * 16 * BASE_FILTERS * 8, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, BASE_FILTERS * 8))(x)
    filters = BASE_FILTERS * 8
    for i in range(4):  # 16 -> 32 -> 64 -> 128 -> 256
        filters //= 2
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same",
                                   name=f"dec_deconv_{i}")(x)
        x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
        x = layers.LeakyReLU(0.2, name=f"dec_act_{i}")(x)
    # Output: 1 channel, no activation (we work in normalised dB space).
    outputs = layers.Conv2D(1, 3, padding="same", name="reconstruction")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


class VAE(keras.Model):
    """End-to-end VAE with reconstruction + KL divergence losses.
    The KL weight (beta) is exposed so train.py can ANNEAL it -- start near 0
    so the model focuses on faithful reconstruction, then ramp up so it
    organises the latent space. This avoids 'posterior collapse' where the
    decoder ignores z and just outputs an average spectrogram."""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="beta")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            recon = self.decoder(z, training=True)
            # Reconstruction loss: per-pixel MSE summed over the image.
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - recon), axis=[1, 2, 3]))
            # KL divergence between N(z_mean, sigma) and N(0, I).
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=False)
        recon = self.decoder(z, training=False)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data - recon), axis=[1, 2, 3]))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


def build_vae():
    """Convenience constructor."""
    return VAE(build_encoder(), build_decoder())


if __name__ == "__main__":
    vae = build_vae()
    vae.encoder.summary()
    vae.decoder.summary()
    print(f"\nLatent dim: {LATENT_DIM} | Input shape: {INPUT_SHAPE}")