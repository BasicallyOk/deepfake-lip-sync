import tensorflow as tf
from tensorflow import keras

from utils.hparams import hparams as hp

class Generator(keras.Model):
    """The auto encoder stack represening the generator"""
    def __init__(self, masked_id_encoder, reference_id_encoder, audio_encoder, decoder):
        super().__init__()
        self.masked_id_encoder = masked_id_encoder
        self.reference_id_encoder = reference_id_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder

    def call(self, x):
        identity, reference, audio = x 
        id_encoding = self.masked_id_encoder(identity)
        ref_encoding = self.reference_id_encoder(reference)
        audio_encoding = self.audio_encoder(audio)
        combined = tf.concat([id_encoding, ref_encoding, audio_encoding], -1) # dimension 0 is batch size
        genenerated_mouth = self.decoder(combined) 
        return tf.concat([identity, genenerated_mouth], axis=-3) # combine generated mouth and original top half

class Discriminator(keras.Model):
    """Deepfake Discriminator"""
    def __init__(self, image_encoder, audio_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
    
    def call(self, x):
        image, audio = x
        image_encoding = self.image_encoder(image)
        audio_encoding = self.audio_encoder(audio)

        # L2-normalize the encoding tensors
        image_encoding = tf.math.l2_normalize(image_encoding, axis=1)
        audio_encoding = tf.math.l2_normalize(audio_encoding, axis=1)

        # measures how much the face matches the audio
        # will return nan without the 1e-12 offset due to https://github.com/tensorflow/tensorflow/issues/12071
        return tf.norm((image_encoding - audio_encoding) + 1e-12, ord='euclidean', axis=1, keepdims=True)
    
class DeepfakeGAN(keras.Model):
    """The Deepfake Generative Adversarial Network model."""
    def __init__(self, generator, discriminator):
        """
        Constructor
        Args:
            generator The generator model
            discriminator: The discriminator model
            test_gen_data: The data to generate a constant human face for the gif
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        """
        Compiles the GAN.
        Args:
            d_optimizer: The optimizer for the discriminator
            g_optimizer: The optimizer for the generator
            d_loss_fn: The loss function for the discriminator
            g_loss_fn: The loss function for the generator
        """
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def call(self, x):
        return self.generator(x)
    
    @tf.function
    def train_step(self, data):
        """
        Trains the GAN for one epoch. Will only train on BATCH_SIZE-1 samples.
        Args:
            data: lists of real images and the audio spectrograms with correspondent indices
        """
        x, original_image = data
        masked_image, ref_image, spectrogram_window, unsync_window = x

        batch_size = tf.shape(original_image)[0]

        # Use the generator to generate images
        generated_images = self.generator((masked_image, ref_image, spectrogram_window))

        train_unsync = tf.random.uniform(()) < 0.5 # 50/50
        if not train_unsync:
            # Add real images to batch
            combined_images = tf.concat(
                [generated_images, original_image], axis=0
                ) 
            combined_audio = tf.concat(
                [spectrogram_window, spectrogram_window], axis=0
                )
        else:
            # Add real images to batch
            combined_images = tf.concat(
                [original_image, original_image], axis=0
                ) 
            combined_audio = tf.concat(
                [unsync_window, spectrogram_window], axis=0
                )
        # Discriminator labels for combined_images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )
        labels += 0.05 * tf.random.uniform(tf.shape(labels)) # Add random noise to the labels

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            predictions = self.discriminator((combined_images, combined_audio))
            d_loss = self.d_loss_fn(labels, predictions)

        # Compute gradients and update weights
        grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            # Use the generator to generate images, must be done again to create connection
            generated_images = self.generator((masked_image, ref_image, spectrogram_window))
            predictions = self.discriminator((generated_images, spectrogram_window))
            # Labels if all real images
            real = tf.zeros((batch_size, 1))
            # Staggering loss function as mse is small
            g_loss = hp.disc_wt *(self.d_loss_fn(real, predictions)) + self.g_loss_fn(original_image, generated_images)

        # Compute gradients and update weights
        grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
                "gen_loss": self.gen_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),
            }
    
    def test_step(self, data):
        x, original_images = data
        masked_image, ref_image, spectrogram_window, _ = x

        batch_size = tf.shape(original_images)[0]

        # inference
        generated_images = self.generator((masked_image, ref_image, spectrogram_window))
        predictions = self.discriminator((generated_images, spectrogram_window))

        # calculate loss
        d_loss = self.d_loss_fn(tf.ones((batch_size, 1)), predictions)
        g_loss = hp.disc_wt *(self.d_loss_fn(tf.zeros((batch_size, 1)), predictions)) + self.g_loss_fn(original_images, generated_images)

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
                "test_gen_loss": self.gen_loss_tracker.result(),
                "test_disc_loss": self.disc_loss_tracker.result(),
            }
