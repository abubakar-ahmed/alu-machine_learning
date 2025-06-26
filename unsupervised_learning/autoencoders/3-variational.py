#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer containing dimensions of the model input
        hidden_layers: list containing number of nodes for each hidden layer
        latent_dims: integer containing dimensions of the latent space

    Returns:
        encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    # Hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Latent space
    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Encoder model
    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=[z_mean, z_log_var, z],
        name='encoder'
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs,
        name='decoder'
    )

    # VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    auto = keras.Model(
        inputs=encoder_inputs,
        outputs=outputs,
        name='autoencoder'
    )

    # Loss function
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        outputs
    )
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - keras.backend.square(z_mean)
    kl_loss -= keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
