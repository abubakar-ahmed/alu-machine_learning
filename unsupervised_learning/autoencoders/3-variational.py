#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """

    # Input layer
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    # Hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Latent space
    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(x)

    # Sampling function using Lambda
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.layers.RandomNormal(mean=0., stddev=1.)(keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Encoder model
    encoder = keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # VAE model
    outputs = decoder(z)
    auto = keras.Model(inputs=encoder_inputs, outputs=outputs)

    # Custom loss using keras.backend
    reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var),
        axis=-1
    )

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
