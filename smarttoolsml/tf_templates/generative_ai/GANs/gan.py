import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from callbacks import ModelMonitor
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     LeakyReLU, Reshape, UpSampling2D)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def predict_with_generator(generator):
    img = generator.predict(np.random.randn(4, 128, 1))
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

    # Loop four times and get images
    for idx, img in enumerate(img):
        ax[idx].imshow(np.squeeze(img))
        ax[idx].title.set_text(idx)


def build_generator():
    model = Sequential()

    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, 4, padding="same", activation="sigmoid"))

    return model


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    return model


class GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)

        # Create attributes for losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fakes images
            y_realfake = tf.concat(
                [tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0
            )

            # Add some noise to the TRUE outputs
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss - BINARYCROSS
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Apply backpropagation - nn learn
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(
                tf.zeros_like(predicted_labels), predicted_labels
            )

        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}


def build_gan():
    generator = build_generator()
    discriminator = build_discriminator()
    g_opt = Adam(learning_rate=0.0001)
    d_opt = Adam(learning_rate=0.00001)
    g_loss = BinaryCrossentropy()
    d_loss = BinaryCrossentropy()
    gan = GAN(generator=generator, discriminator=discriminator)
    gan.compile(g_opt, d_opt, g_loss, d_loss)
    return gan


def train_gan(gan, dataset, epochs, callbacks):
    """_summary_

    Args:
        gan (_type_): _description_
        dataset (_type_): _description_
        epochs (_type_): _description_
        callbacks (_type_): _description_

    Example usage:
        callbacks = [ModelMonitor()]

        train_gan(gan, dataset, epochs, callbacks)
    """
    history = gan.fit(dataset, epochs=epochs, callbacks=callbacks)
    return history
