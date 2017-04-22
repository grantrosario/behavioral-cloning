import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

"""
CONSTANTS
"""

# Data augmentation constants
TRANS_X_RANGE = 100  # Number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2)
TRANS_Y_RANGE = 40  # Number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2)
TRANS_ANGLE = .3  # Maximum angle change when translating in the X direction
OFF_CENTER_IMG = .25  # Angle change when using off center images

BRIGHTNESS_RANGE = .25  # The range of brightness changes
ANGLE_THRESHOLD = 1.  # The maximum magitude of the angle possible

# Training constants
BATCH = 128  # Number of images per batch
TRAIN_BATCH_PER_EPOCH = 160  # Number of batches per epoch for training
EPOCHS = 5  # Minimum number of epochs to train the model on

# Image constants
IMG_ROWS = 64  # Number of rows in the image
IMG_COLS = 64  # Number of cols in the image
IMG_CH = 3  # Number of channels in the image

# Correction angle
COR_ANGLE = 0.2


# Data Augmentation
def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    # Remove the unwanted top scene and retain only the track
    crop = img[60:140, :, :]

    # Resize the image
    resize = cv2.resize(crop, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)

    return resize

def img_change_brightness(img):
    # Convert the image to HSV
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply to the image
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)


def img_translate(img, x_translation):
    # Randomly compute a Y translation
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)

    # Form the translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    # Translate the image
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

def data_augment(img_path, angle):
    """
    Augments the data by generating new images based on the base image found in img_path
    :param img_path: Path to the image to be used as the base image
    :param angle: The steering angle of the current image
    :param threshold: If the new angle is below this threshold, then the image is dropped
    :return:
        new_img, new_angle of the augmented image / angle (or)
        None, None if the new angle is below the threshold
    """
    # Randomly form the X translation distance and compute the resulting steering angle change
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE

    # Let's read the image
    img = cv2.imread(img_path)  # Read in the image
    img = img_change_brightness(img)  # Randomly change the brightness
    img = img_translate(img, x_translation)  # Translate the image in X and Y direction
    if np.random.randint(2) == 0:  # Flip the image
        img = np.fliplr(img)
        new_angle = -new_angle
    img = img_pre_process(img)  # Pre process the image

    return img, new_angle


def generator(samples, batch_size):
    num_samples = len(samples)
    _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
    _y = np.zeros(BATCH, dtype=np.float)
    out_idx = 0
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])

                img_choice = np.random.randint(3)
                if img_choice == 0:
                    img_path = 'my_data/IMG/' + batch_sample[1].split('/')[-1] # left image
                    angle += COR_ANGLE

                elif img_choice == 1:
                    img_path = 'my_data/IMG/' + batch_sample[0].split('/')[-1] # center image

                else:
                    img_path = 'my_data/IMG/' + batch_sample[2].split('/')[-1] # right_image
                    angle -= COR_ANGLE

                img, angle = data_augment(img_path, angle)

                # Check if we've got valid values
                if img is not None:
                    _x[out_idx] = img
                    _y[out_idx] = angle
                    out_idx += 1

                # Check if we've enough values to yield
                if out_idx >= BATCH:
                    yield _x, _y

                    # Reset the values back
                    _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
                    _y = np.zeros(BATCH, dtype=np.float)
                    out_idx = 0


lines = []
with open('my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, BATCH)
validation_generator = generator(validation_samples, BATCH)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(IMG_ROWS, IMG_COLS, IMG_CH)))
model.add(Convolution2D(24,5,5, border_mode='valid', subsample=(2,2),activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(36,5,5, border_mode='valid', subsample=(2,2),activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(48,5,5, border_mode='valid', subsample=(2,2),activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(64,3,3, border_mode='same', subsample=(2,2),activation="relu", W_regularizer=l2(0.001)))
model.add(Convolution2D(64,3,3, border_mode='valid', subsample=(2,2),activation="relu", W_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, W_regularizer=l2(0.001)))

adam = Adam()
model.compile(loss = 'mse', optimizer = adam)
history_object = model.fit_generator(train_generator, samples_per_epoch = TRAIN_BATCH_PER_EPOCH * BATCH,
                   nb_epoch = EPOCHS, validation_data = validation_generator,
                   nb_val_samples = BATCH, verbose = 1)


model.save("model.h5")
print("Saved model to disk")
exit()
