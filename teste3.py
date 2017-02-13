__author__ = 'Ibis'

#(72, 288, 3)
model_height = 64
model_weight = 64

'''
    Read driving_log.csv and save it in a vector with the following configuration:
        * Image path
        * Steering angle
        * Throttle
        * Brake
        * Speed
'''

import csv
import numpy as np

list_images = list()

# The following value is added and subtracted from the steering angle for the images of the right and left side of the car
offset = 0.2 # Best value: 0.20 0.22

import random
bias = 1 # Best value: 0.60
path = "new_data"

with open(path + '\driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        steering = float(row[3])
        throttle = float(row[4])
        brake = float(row[5])
        speed = float(row[6])

        steering_thresh = np.random.rand()
        if (abs(steering) + bias) < steering_thresh:
            pass # drop this sample
        else:
            if (steering == 0):
                if (np.random.rand() > 0): # Best value: 0.75 # Mudar para 0.8
                    # Center image
                    list_images.append([row[0].replace(" ", ""), steering, throttle, brake, speed])
                    # # Left image
                    # list_images.append([row[1].replace(" ", ""), steering + offset, throttle, brake, speed])
                    # # Right image
                    # list_images.append([row[2].replace(" ", ""), steering - offset, throttle, brake, speed])
            else:
                # Center image
                list_images.append([row[0].replace(" ", ""), steering, throttle, brake, speed])
                # # Left image
                # list_images.append([row[1].replace(" ", ""), steering + offset, throttle, brake, speed])
                # # Right image
                # list_images.append([row[2].replace(" ", ""), steering - offset, throttle, brake, speed])

print('Images mapped with {} examples'.format(len(list_images)))

import matplotlib.pyplot as plt

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import random

def plot_image_prob(image):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    ax.imshow(image, vmin=0, vmax=255)
    plt.axis('off')
    ax.axis('tight')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

def flip_image(img, steering):
    if random.randint(0, 1):
        return cv2.flip(img, 1), -steering
    else:
        return img, steering

def brightness_image(img, steering):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.4, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), steering

def rotate_image(img, steering):
    rows,cols,channel = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), random.uniform(-5, 5), 1)
    return cv2.warpAffine(img,M,(cols,rows)), steering

def cut_image(img):
    rows,cols,channel = img.shape
    top = int(.4 * rows)
    botton = int(.85 * rows)
    #border = int(.05 * cols)
    return img[top:botton, :, :] #border:cols-border, :]

def translate_image(img, steering, horz_range=30, vert_range=5):
    rows, cols, chs = img.shape
    tx = np.random.randint(-horz_range, horz_range+1)
    ty = np.random.randint(-vert_range, vert_range+1)
    steering = steering + tx * 0.003 # mul by steering angle units per pixel
    tr_M = np.float32([[1,0,tx], [0,1,ty]])
    img = cv2.warpAffine(img, tr_M, (cols,rows), borderMode=1)
    return img, steering

def shadow_image(img, steering):
    rows, cols, chs = img.shape

    # Generate a separate buffer
    shadows = img.copy()

    randomUp = int(random.random() * cols)
    randomDown = int(random.random() * cols)

    if random.randint(0, 1):
        poly = [[randomUp,0],[cols,0],[cols,rows], [randomDown,rows]]
    else:
        poly = [[randomUp,0],[0,0],[0,rows], [randomDown,0]]

    cv2.fillPoly(shadows, np.array([poly]), -1)

    alpha = np.random.uniform(0.6, 0.9)
    return cv2.addWeighted(shadows, alpha, img, 1-alpha,0,img), steering

import progressbar
import pickle
import os.path

def load_data_batch(list_images, indices):
    # Placeholders for the images and labels from web
    X = np.empty(shape = [1, 320, 160, 3], dtype = np.uint8)
    y = np.empty(shape = [1], dtype = np.float32)

    for i in indices:
        image = Image.open(list_images[i][0])
        steering = list_images[i][1]

        X = np.vstack([X, np.reshape(image, [1, 320, 160, 3])])
        y = np.vstack([y, steering])

    # Get rid of the first empty row
    X = X[1:, :, :, :]
    y = y[1:]

    return X, y

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout, Lambda
from keras.optimizers import Adam, Nadam
from keras.layers.normalization import BatchNormalization

model = Sequential()

# # Layer 1: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
# model.add(Convolution2D(24, 5, 5, input_shape=(72, 288, 3), border_mode='valid', init='normal'))
# model.add(Activation('elu'))
# model.add(MaxPooling2D((2, 2)))
#
# # Layer 2: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
# model.add(Convolution2D(36, 5, 5, border_mode='valid', init='normal'))
# model.add(Activation('elu'))
# model.add(MaxPooling2D((2, 2)))
#
# # Layer 3: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
# model.add(Convolution2D(48, 5, 5, border_mode='valid', init='normal'))
# model.add(Activation('elu'))
# #model.add(MaxPooling2D((2, 2)))
#
# # Layer 4: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
# model.add(Convolution2D(64, 3, 3, border_mode='valid', init='normal'))
# model.add(Activation('elu'))
# #model.add(MaxPooling2D((2, 2)))
#
# # Layer 5: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
# model.add(Convolution2D(64, 3, 3, border_mode='valid', init='normal'))
# model.add(Activation('elu'))
# #model.add(MaxPooling2D((2, 2)))
#
# model.add(Flatten())
#
# model.add(Dense(1164, init='normal'))
# model.add(Dropout(p=0.5))
# model.add(Activation('elu'))
#
# model.add(Dense(100, init='normal'))
# model.add(Dropout(p=0.5))
# model.add(Activation('elu'))
#
# model.add(Dense(50, init='normal'))
# model.add(Dropout(p=0.5))
# model.add(Activation('elu'))
#
# model.add(Dense(10, init='normal'))
# model.add(Dropout(p=0.5))
# model.add(Activation('elu'))
#
# model.add(Dense(1, init='normal'))
#


init = 'normal'
input_shape=(model_height, model_weight, 3)

model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape))

model.add(Convolution2D(16, 3, 3, border_mode='valid', init=init))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3, border_mode='valid', init=init))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Convolution2D(48, 3, 3, border_mode='valid', init=init))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(256, init=init))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(128, init=init))
model.add(Activation('elu'))

model.add(Dense(16, init=init))
model.add(Activation('elu'))

model.add(Dense(1, init=init))

print(model.summary())

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Shuffle list
list_images = shuffle(list_images)

# Split testing set
train_set, valid_set = train_test_split(list_images, test_size=0.2, random_state=0)

print("Train set =", len(train_set))
print("Validation set =", len(valid_set))

# # Shuffle list
# X, y = shuffle(X, y)

# # Split testing set
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# print("X_train shape =", X_train.shape)
# print("y_train shape =", y_train.shape)
# print("X_valid shape =", X_valid.shape)
# print("y_valid shape =", y_valid.shape)

def myGenerator(list, batch_size, samples_epoch, flag="test"):
    while 1:
        list = shuffle(list)

        indices = random.sample(range(len(list)), batch_size)

        X_batch = np.empty(shape = [1, model_height, model_weight, 3], dtype = np.uint8)
        y_batch = np.empty(shape = [1], dtype = np.float32)

        X, y = load_data_batch(list, indices)

        for i in range(0, batch_size):

            image = X[i]
            steering = y[i]

            if (flag == "test"):
                image, steering = brightness_image(np.copy(image), steering)
                #image, steering = rotate_image(np.copy(image), steering)
                image, steering = translate_image(np.copy(image), steering)
                image, steering = shadow_image(np.copy(image), steering)
                image, steering = flip_image(np.copy(image), steering)

            image = cut_image(image)

            image = cv2.resize(image,(model_height, model_height), interpolation = cv2.INTER_CUBIC)

            X_batch = np.vstack([X_batch, np.reshape(image, [1, model_height, model_weight, 3])])
            y_batch = np.vstack([y_batch, steering])

        # Get rid of the first empty row
        X_batch = X_batch[1:, :, :, :]
        y_batch = y_batch[1:]

        yield (X_batch, y_batch)
#
# def myGenerator2(list, batch_size, samples_epoch, flag="test"):
#     while 1:
#         list = shuffle(list)
#
#         indices = random.sample(range(len(list)), samples_epoch)
#
#         for j in range(0, samples_epoch, batch_size):
#
#             X_batch = np.empty(shape = [1, model_height, model_weight, 3], dtype = np.uint8)
#             y_batch = np.empty(shape = [1], dtype = np.float32)
#
#             X, y = load_data_batch(list, indices[j:j+batch_size])
#
#             for i in range(0, batch_size):
#
#                 image = X[i]
#                 steering = y[i]
#
#                 if (flag == "test"):
#                     image, steering = brightness_image(np.copy(image), steering)
#                     image, steering = rotate_image(np.copy(image), steering)
#                     image, steering = translate_image(np.copy(image), steering)
#                     image, steering = shadow_image(np.copy(image), steering)
#                     image, steering = flip_image(np.copy(image), steering)
#
#                 image = cut_image(image)
#
#                 #image = cv2.resize(image,(64, 64), interpolation = cv2.INTER_CUBIC)
#
#                 X_batch = np.vstack([X_batch, np.reshape(image, [1, model_height, model_weight, 3])])
#                 y_batch = np.vstack([y_batch, steering])
#
#             # Get rid of the first empty row
#             X_batch = X_batch[1:, :, :, :]
#             y_batch = y_batch[1:]
#
#             yield (X_batch, y_batch)
# generator = myGenerator(X_valid, y_valid, 40)
# a, b = next(generator)#generator.fit()
# print(a.shape)
# print(b.shape)

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# compile and fit model
print("Fitting model")
model.compile(loss='mse', metrics=['mse'], optimizer=Nadam(lr=0.001))

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=3, verbose=1)
#model_checkpoint = ModelCheckpoint(filepath='model.weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
learning_rate_plateau_reducer = ReduceLROnPlateau(verbose=1, patience=2, epsilon=1e-5)

batch_size=400
samples_epoch = 4000
n_epoch = 8

fit = model.fit_generator(myGenerator(train_set, batch_size, samples_epoch),
                          verbose=1, samples_per_epoch=samples_epoch,
                          nb_epoch=n_epoch,
                          callbacks=[learning_rate_plateau_reducer],
                          validation_data=myGenerator(valid_set, 100, 1000, "validation"),
                          nb_val_samples = 1000)

# scores = model.evaluate(myGenerator(X_valid, y_valid, batch_size, "validation"), verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# compare model predicted steering angles with labeled values
# y_train_predict = model.predict(X_train)
# print(y_train_predict.shape)

# evaluate the model
# np.set_printoptions(suppress=True)
# print(y_train_predict[0:40].T)
# print(y_train[0:40].T)

# output model
print("Saving model structure and weights")
model_json = model.to_json()
import json
with open ('model.json', 'w') as f:
    json.dump(model_json, f, indent=4, sort_keys=True, separators=(',', ':'))

model.save_weights('model.h5')

import gc; gc.collect()