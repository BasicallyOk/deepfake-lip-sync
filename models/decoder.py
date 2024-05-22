from tensorflow import keras

# verify the latent dimensions using the models
latent_dims = 1024+512+1024
image_decoder = keras.models.Sequential([
    # the comma makes tensorflow interpret latent_dims as a valid shape
    keras.layers.Input(shape=(latent_dims, ), name="deepfake_encoding"),
    keras.layers.Reshape((4, 4, 160)), # 4x4x160 = 2560
    keras.layers.Conv2DTranspose(128, 3, (1, 2), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(16, 3, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(3, 1, 1, padding='same', activation='sigmoid'),
], name='face_decoder')