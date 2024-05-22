from tensorflow import keras

IMAGE_SHAPE = (64, 64, 3)
AUDIO_SPECTROGRAM_SHAPE = (4, 601, 1)
MASKED_IMAGE_SHAPE = (32, 64, 3)

identity_encoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=IMAGE_SHAPE, name="ref_image_input"),
    keras.layers.GaussianNoise(0.1), # introduce some noise
    keras.layers.Conv2D(16, 7, 4, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='id_encoding'),
], name="id_encoder")

identity_encoder_disc = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=IMAGE_SHAPE, name="image_input"),
    keras.layers.Conv2D(16, 7, 4, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 5, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, 1, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='id_encoding_disc'),
], name="id_encoder_disc")

masked_id_encoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=MASKED_IMAGE_SHAPE, name="masked_image_input"),
    keras.layers.Conv2D(16, 7, 4, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 5, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 1, 1, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='masked_id_encoding'),
], name="masked_id")

audio_encoder = keras.models.Sequential([
    keras.layers.Input(shape=AUDIO_SPECTROGRAM_SHAPE, name="audio_input"),
    keras.layers.Conv2D(16, 7, (2, 6), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 5, (1, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, (1, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, 3, (1, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='audio_encoding'),
], name="audio_encoder")

# sigmoid ensures euclidean distance is in [0,1]
audio_encoder_disc = keras.models.Sequential([
    keras.layers.Input(shape=AUDIO_SPECTROGRAM_SHAPE, name="audio_input"),
    keras.layers.Conv2D(16, 7, (2, 6), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 5, (1, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, 3, (1, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, 3, (1, 3), padding='same', activation='sigmoid'), 
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(name='audio_encoding_disc'),
], name="audio_encoder_disc")