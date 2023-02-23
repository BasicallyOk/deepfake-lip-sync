"""
Authors: Huy Nguyen, Yu Xuan Liu
Maintainer: Khoa Nguyen
"""
# import utils.get_face_from_video as get_face_from_video
import numpy as np   # linear algebra
import cv2
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from keras import layers, models
import re

"""
Visual Quality Discriminator Model Definition
Used to discriminate (hah, get it?) between real and fake images
Meant to be used as the discriminator
"""
def quality_discriminator(training=True):
    """
    The quality discriminator in all of its glory
    Going to be a combined model of both audio and video.
    The one below now is current the video part.
    """
    # Define image encoding stack
    image_input = keras.Input(shape=(256, 256, 3), name="input_face_disc")

    x = layers.Conv2D(8, 7, 2, padding="same", activation='relu', input_shape=(256, 256, 3))(image_input)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2D(16, 5, 2, padding="same", activation='relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2D(32, 3, 2, padding="same", activation='relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2D(64, 3, 2, padding="same", activation='relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2D(64, 3, 2, padding="same", activation='sigmoid')(x)

    image_encoding = layers.Flatten()(x)

    # Define audio encoding stack
    audio_input = keras.Input(shape=(6, 513, 1), name="input_audio_disc")

    y = layers.Conv2D(16, 7, (1, 3), padding="same", activation='relu', input_shape=(6, 513, 1))(audio_input)
    y = tfa.layers.InstanceNormalization()(y)
    y = layers.Conv2D(32, 5, (1, 2), padding="same", activation='relu')(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = layers.Conv2D(64, 5, (1, 3), padding="same", activation='relu')(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = layers.Conv2D(64, 5, (1, 3), padding="same", activation='relu')(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = layers.Conv2D(128, 3, 1, padding="valid", activation='sigmoid')(y)

    audio_encoding = layers.Flatten()(y)

    # L2-normalize the encoding tensors
    image_encoding = tf.math.l2_normalize(image_encoding, axis=1)
    audio_encoding = tf.math.l2_normalize(audio_encoding, axis=1)

    # Find euclidean distance between image_encoding and audio_encoding
    # Essentially trying to detect if the face is saying the audio
    # Will return nan without the 1e-12 offset due to https://github.com/tensorflow/tensorflow/issues/12071
    d = tf.norm((image_encoding - audio_encoding) + 1e-12, ord='euclidean', axis=1, keepdims=True)

    discriminator = keras.Model(inputs=[image_input, audio_input], outputs=[d], name="discriminator")

    discriminator.compile(loss=tfa.losses.ContrastiveLoss(), optimizer='adam')
    return discriminator

"""
Everything from here onwards is for testing purposes only
"""

# All lists below are supposed to be lists of numpy arrays.
x_train = []  # raw_videos set
y_train = []  # raw_videos label

x_test = []   # test set
y_test = []   # test label

def append_train(x_train, y_train, img_path: str, label: str):
    """
    Add image and label to raw_videos lists
    Parameters:
        img_path: the path to the image
        label: the label of the image
    """
    img = cv2.imread(img_path)
    # img_np = np.asarray(img)
    x_train.append(img)
    # Categorical Data time.
    if label == "FAKE":
        y_train.append(0)
    elif label == "REAL":
        y_train.append(1)


def append_test(x_test, y_test, img_path, label: str):
    """
    Add image and label to test lists
    Parameters:
        img_path: the path to the image
        label: the label of the image
    """
    # Note to self, if you want to run get_face_from_video.py on the same set of videos, delete the entire dataset and
    # run it again.
    img = cv2.imread(img_path)
    x_test.append(img)
    # Categorical Data time.
    if label == "FAKE":
        y_test.append(0)
    elif label == "REAL":
        y_test.append(1)


def load_dataset(data_path: str):
    """
    Given a path with the structure
    dataset
    |---test
    |   |---real
    |   |---fake
    |---train
    |   |---real
    |   |---fake
    Load data into its respective python list
    Assumes a UNIX-based file system
    """
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    counter = 0
    
    # Load train data
    for img in os.listdir(os.path.join(train_path, "real")):
        counter += 1
        if counter == 3000:
            break
        img_path = os.path.join(train_path, "real", img)
        append_train(x_train, y_train, img_path, "REAL")
    # Reset the value of counter
    counter = 0
    for img in os.listdir(os.path.join(train_path, "fake")):
        counter += 1
        if counter == 3000:
            break
        img_path = os.path.join(train_path, "fake", img)
        append_train(x_train, y_train, img_path, "FAKE")
    counter = 0
    # Load test data
    for img in os.listdir(os.path.join(test_path, "real")):
        counter += 1
        if counter == 1000:
            break
        img_path = os.path.join(test_path, "real", img)
        append_test(x_test, y_test, img_path, "REAL")
    counter = 0
    for img in os.listdir(os.path.join(test_path, "fake")):
        counter += 1
        if counter == 1000:
            break
        img_path = os.path.join(test_path, "fake", img)
        append_test(x_test, y_test, img_path, "FAKE")

    return x_train, y_train, x_test, y_test

#
# Helper function to show a list of images with their relating titles
#


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()

#
# Show some random training and test images
#


def choose_imgs_and_plot():
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, len(x_train))
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, len(x_test))
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))
    show_images(images_2_show, titles_2_show)


def get_path(script_path):
    """
    Get the path to the current script. Crop it out to get the path of the project.
    Assume that the project name is deepfake-lip-sync.
    :return the absolute path of the project.
    """
    if os.name == "nt":
        # Unreachable?
        pattern = r"^(.*\\\\deepfake-lip-sync).*"
    else:
        pattern = r"(.*/deepfake-lip-sync).*"
    match = re.match(pattern=pattern, string=script_path)
    return match[1]

def pretrained_quality_discriminator(project_path):
    path = os.path.join(project_path, "saved_models")
    dirs = os.listdir(path)
    if len(dirs) == 0:          # turn != to == later.
        print("No models can be found. If you are trying to train, use quality_discriminator instead.")
        return None
    else:
        # Just get the first one available I think
        model_path = os.path.join(path, "huy")    # Change dirs[0] to the model of whom you want to load into your
                                                  # model.
        print(f"Loading the model from {model_path}")
        old_model = tf.keras.models.load_model(model_path)
        old_model.summary()
        return old_model


# save the checkpoint to the file path...
# So, does checkpoint get activated after every epoch, or does it get activated after the step?
# First of all, how to put the save checkpoint code while fitting?
# Second of all, how to restore check points and continuing running from there.
#               Is there a way to indicate that the checkpoint has been restored?

def restore_checkpoint(model, ckpt, manager, epochs, save_checkpoint_path):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        number = re.match(pattern=r".*(.*(\d)$)", string=str(manager.latest_checkpoint))[1]
        print(f"Restored from {manager.latest_checkpoint}")
        return int(number)
    else:
        print("Intitialize from scratch since no checkpoint is found..")
        return 0


def test_train():
    project_path = get_path(os.path.dirname(__file__))
    dataset_path = os.path.join(project_path, "dataset")
    x_train, y_train, x_test, y_test = load_dataset(dataset_path)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    x_train, y_train = unison_shuffled_copies(x_train, y_train)

    print(f"x_train: {x_train.shape}, x_test {x_test.shape}")
    print(f"y_train: {y_train.shape}, y_test {y_test.shape}")

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model = quality_discriminator()
    #model = quality_discriminator_pretrained()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mean_absolute_error'])

    save_checkpoint_path = "saved_checkpoints"
    saved_model_path = os.path.join(project_path, "saved_models", "huy")
    save_checkpoint_path = os.path.join(project_path, save_checkpoint_path)
    epochs = 3

    # model.save(saved_model_path)   # TODO: get dotenv working and make this an env variable.
    # Comment: I suppose that we won't need the line above then. But I'm gonna comment it out just in case.
    # Maybe only use the checkpoint if there is no saved_models.
    # If there is, check that there is no checkpoints there, and if there is no checkpoints, then use restore the models
    # Ok, so, extract the number at the checkpoint file.
    ckpt = tf.train.Checkpoint(model)
    manager = tf.train.CheckpointManager(ckpt, save_checkpoint_path, max_to_keep=epochs)

    epochs_passed = restore_checkpoint(model, ckpt, manager, epochs, save_checkpoint_path)
    num_loops = epochs - epochs_passed
    print(f"Total epochs: {epochs}\nNumber of epochs gonna be run this session: {num_loops}")
    for epoch in range(num_loops):
        print(f"Epoch {epoch} starting now:")
        model.fit(x_train, y_train)
        manager.save()

    # delete all checkpoints after the entire run is complete.
    for file in os.listdir(save_checkpoint_path):
        os.remove(os.path.join(save_checkpoint_path, file))


if __name__ == "__main__":
    model = quality_discriminator()
    model.summary()
