import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras import models
from moviepy.editor import *
from PIL import Image
from utils.audio_spectrogram import stft

from keras import layers
from utils import get_face_from_video

# In the future. Maybe try changing the input shape to basically use longer audio files. Right now it can only
# use the one with the duration of  1/30 second where 30 is the fps of the data used.
WEIGHT_INIT_STDDEV = 0.02

"""
    PLACEHOLDER.....
    Generates a number
    Input is a random seed of size (100,)
    The generator is supposed to take
    
    
    
    
    image and also an audio frame extracted (its gonna match the frame from the video)
    
    we would cut off the bottom half of the image (its gonna be filled in with black)
    
    and then we use the generator 
"""


def identity_encoder():
    """A model for receiving the image as a numpy array."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(256, 256, 6), name="image_input"),
        layers.Conv2D(16, 1, 1, activation='relu'),
        layers.MaxPool2D(32, 2, padding='same'),
        layers.Conv2D(32, 5, 2, activation='relu'),
        layers.MaxPool2D(64, 2, padding='same'),
        layers.Conv2D(60, 7, 2, activation='relu'),
        layers.MaxPool2D(64, 1, padding='same'),
        layers.Conv2D(61, 7, 1, activation='sigmoid', padding='valid'),
        tf.keras.layers.Flatten(name='identity_encoding'),
    ], name="image")
    return model


def audio_encoder():
    """A model for receiving the audio as a numpy array."""
    loudness = 80  # Probably the decibel of the sound. Not sure how it's like that. tbf, I actually have no clue if the variable name is c
    # correct for this value.
    step_size = 34  # Presumably the time each frame occupies. So a frame can last like 1/30th of a second if the video is in 30FPS.

    shape_of_audio_np = (6, 513, 1)

    num_labels = 64
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    # norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))  May have to remove this.

    model = models.Sequential([
        layers.Input(shape=shape_of_audio_np, name="audio_input"),
        # Normalize.
        norm_layer,
        layers.Conv2D(3, 7, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(9, 5, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(30, 3, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(123, 1, 1, activation='sigmoid', padding='valid'),
        layers.Flatten(name='audio_encoding'),
    ], name="audio")
    # model.summary()
    return model


def generator():
    """
    Returns the function (in discrete sense) that is a model.
    """
    identity_model = identity_encoder()
    audio_model = audio_encoder()

    # x = layers.Concatenate([identity_model.layers[-1].output, audio_model.layers[-1].output])
    x = tf.concat([identity_model.output, audio_model.output], axis=-1)
    # I just copied it from the Simpson one. The parameters, to be exact. And I have no clue what the parameters are.
    # combined_output = layers.Conv2DTranspose(filters=3, kernel_size=[5, 5], strides=[1, 1], padding="SAME",
    #                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV),
    #                                          name="logits")(x)
    combined_output = layers.Dense(16*16*16, use_bias=False)(x)
    combined_output = layers.BatchNormalization()(combined_output)
    # combined_output = layers.BatchNormalization()(x)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Reshape((16, 16, 16))(combined_output)
    # assert tf.shape(combined_output) == (None, 16, 16, 768)  # Note: None is the batch size

    combined_output = layers.Conv2DTranspose(18, (5, 5), strides=(2, 2), padding='same', use_bias=False)(
        combined_output)
    # assert combined_output.output_shape == (None, 32, 32, 192)
    # assert tf.shape(combined_output) == (None, 32, 32, 192)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Conv2DTranspose(12, (5, 5), strides=(2, 2), padding='same', use_bias=False)(
        combined_output)
    # assert tf.shape(combined_output) == (None, 128, 128, 12)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Conv2DTranspose(6, (5, 5), strides=(4, 4), padding='same', use_bias=False)(
        combined_output)
    # assert tf.shape(combined_output) == (None, 128, 128, 12)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)

    combined_output = layers.Conv2DTranspose(3, (5, 5), strides=1, padding='same', use_bias=False)(combined_output)
    combined_output = layers.BatchNormalization()(combined_output)
    combined_output = layers.LeakyReLU()(combined_output)
    # assert tf.shape(combined_output) == (None, 256, 256, 3)

    combined_output = layers.Conv2D(3, kernel_size=1, strides=1, padding="same", activation="sigmoid")(combined_output)

    combined_model = keras.Model(inputs=[identity_model.input, audio_model.input],
                                 outputs=[combined_output], name="combined_model")
    # combined_model.summary()

    combined_model.compile(
        optimizer='adam',
        loss='mae'
    )
    return combined_model

def combined_generator(generator, discriminator):
    """
    The generator will now receive the discriminator output to train itself, hence the word "combined"
    Implemented using Tensorflow's symbolic tensors
    """
    # This is the start of the ultimate generator
    # Input is just input. At this stage, they are separate from the whole model.
    input_face = keras.Input(shape=(256, 256, 6), name="input_face_comb")
    input_audio = keras.Input(shape=(6, 513, 1), name="input_audio_comb")

    # This can be considered as the "body" of the ultimate generator, gen is a function.

    # Since gen is a function (in discrete sense), it can receive arguments. fake_face is the output of the function.
    fake_face = generator([input_face, input_audio])
    discriminator.trainable = False
    # The discriminator is also a function, so it can receive arguments. d is the output of the function.
    d = discriminator([fake_face, input_audio])

    # This is the ultimate generator which is also a function, starting with the inputs, and the outputs are outputs of both 
    # the generator and the discrinator's outputs
    gan_generator_model = keras.Model([input_face, input_audio], [fake_face, d])

    # Now the model is compiled.
    gan_generator_model.compile(loss=['mae', tfa.losses.ContrastiveLoss()], 
                            optimizer='adam', loss_weights=[1., .01])

    return gan_generator_model


# mask_image("/home/hnguyen/PycharmProjects/deepfake-lip-sync/dataset/train/fake/FAKE_aktnlyqpah.mp4_111.png")


def extract_audio():
    """
    
    """
    project_path = get_face_from_video.get_path(os.path.dirname(__file__))
    ds_path = os.path.join(project_path, "dataset")
    raw_data_path = os.path.join(project_path, "raw_videos")
    print(project_path, ds_path, raw_data_path)

    # Get the filepaths and the metadata of it.
    file_paths, meta_paths = get_face_from_video.get_files_and_get_meta_file(raw_data_path)

    my_clip = VideoFileClip(r"C:\Users\Yxliu\OneDrive\Documents\dfdc_train_part_1\aassnaulhq.mp4")

    frame_time = 1 / my_clip.fps
    for i in range(0, int(my_clip.duration), frame_time):
        my_clip = my_clip.subclip(i, i + frame_time)
        audio = my_clip.audio
        audio.preview()


def test_generate():
    print("testing")
    img = Image.open("/home/hnguyen/PycharmProjects/deepfake-lip-sync/dataset/train/fake/FAKE_aahsnkchkz_125.png")
    seed_1 = np.asarray(img)
    filepath = f"/home/hnguyen/PycharmProjects/deepfake-lip-sync/utils/audio/vmigrsncac_audio_132.wav"
    samplerate, samples = wav.read(filepath)

    seed_2 = stft(samples, 2 ** 10)
    seed_2 = np.reshape(seed_2, newshape=(6, 513, 1))
    combined_inputs = {"image_input": np.expand_dims(seed_1, axis=0),
                       "audio_input": np.expand_dims(seed_2, axis=0)}
    print(combined_inputs["image_input"].shape)
    print(combined_inputs["audio_input"].shape)

    combined = generator()
    generated_img = combined(combined_inputs, training=False)
    print(generated_img.shape)
    # print(generated_img[0].shape)
    print("done.")
    plt.imshow(generated_img[0])


if __name__ == "__main__":
    id_encoder = identity_encoder()
