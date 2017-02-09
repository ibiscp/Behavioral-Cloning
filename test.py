__author__ = 'Ibis'

#-----------------------------------------------------------------
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
offset = 0.2

import random
bias = 0

with open('data\driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        steering = float(row[3])
        throttle = float(row[4])
        brake = float(row[5])
        speed = float(row[6])

        if (steering == 0):
            if (np.random.rand() > 0.99):
                # Center image
                list_images.append([row[0].replace(" ", ""), steering, throttle, brake, speed])
        else:
            if (np.random.rand() > 0.0):
                steering_thresh = np.random.rand()
                if (abs(steering) + bias) > steering_thresh:
                    # Center image
                    list_images.append([row[0].replace(" ", ""), steering, throttle, brake, speed])
                    # Left image
                    list_images.append([row[1].replace(" ", ""), steering + offset, throttle, brake, speed])
                    # Right image
                    list_images.append([row[2].replace(" ", ""), steering - offset, throttle, brake, speed])

print('Images mapped with {} examples'.format(len(list_images)))

#--------------------------------------------------------------

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
    #if random.randint(0, 1):
    return cv2.flip(img, 1), -steering
    #else:
    #    return img, steering

def brightness_image(img, steering):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.3, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), steering

def rotate_image(img, steering):
    rows,cols,channel = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), random.uniform(-5, 5), 1)
    return cv2.warpAffine(img,M,(cols,rows)), steering

def cut_image(img):
    rows,cols,channel = img.shape
    top = int(.4 * rows)
    botton = int(.85 * rows)
    border = int(.05 * cols)
    return img[top:botton, border:cols-border, :]

def translate_image(img, steering, horz_range=30, vert_range=5):
    rows, cols, chs = img.shape
    tx = np.random.randint(-horz_range, horz_range+1)
    ty = np.random.randint(-vert_range, vert_range+1)
    steering = steering + tx * 0.004 # mul by steering angle units per pixel
    tr_M = np.float32([[1,0,tx], [0,1,ty]])
    img = cv2.warpAffine(img, tr_M, (cols,rows), borderMode=1)
    return img, steering

#-------------------------------------------------------------

import progressbar
import pickle
import os.path

def load_data(list_images):
    # Placeholders for the images and labels from web
    X = np.empty(shape = [1, 320, 160, 3], dtype = np.uint8)
    y = np.empty(shape = [1], dtype = np.float32)

    bar = progressbar.ProgressBar()
    for i in bar(range(len(list_images))):
        image = Image.open("data/" + list_images[i][0])
        steering = list_images[i][1]

        X = np.vstack([X, np.reshape(image, [1, 320, 160, 3])])
        y = np.vstack([y, steering])

        image, steering = flip_image(np.copy(image), steering)

        X = np.vstack([X, np.reshape(image, [1, 320, 160, 3])])
        y = np.vstack([y, steering])

    # Get rid of the first empty row
    X = X[1:, :, :, :]
    y = y[1:]

    return X, y

file = "data.p" # File name

# If file exists read, if not, create file
if (os.path.isfile(file)):
    with open(file, mode='rb') as f:
        data = pickle.load(f)

    X, y = data['images'], data['steering']
    print("File already in disk! Images loaded\n")
else:
    print("Creating image file")
    X, y = load_data(list_images)

    # Save file to
    d = {'images': X, 'steering': y}

    output = open('data.p', 'wb')
    pickle.dump(d, output)
    output.close()

# Convert data to float
X = X.astype('float32')
y = y.astype('float32')

print("X_train shape =", X.shape)
print("y_train shape =", y.shape)

#-----------------------------------------------------------------------

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Shuffle list
# X, y = shuffle(X, y)

# Split testing set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train shape =", X_train.shape)
print("y_train shape =", y_train.shape)
print("X_valid shape =", X_valid.shape)
print("y_valid shape =", y_valid.shape)

#----------------------------------------------------------------------------

def myGenerator(X, y, batch_size, flag="test"):
    while 1:
        X, y = shuffle(X, y)

        for batch_ind in range(0, len(X), batch_size):

            X_batch = np.empty(shape = [1, 288, 72, 3], dtype = np.uint8)
            y_batch = np.empty(shape = [1], dtype = np.float32)

            for i in range(batch_ind, min(batch_ind + batch_size, len(X))):

                image = X[i]
                steering = y[i]

                image, steering = brightness_image(np.copy(image), steering)
                image, steering = rotate_image(np.copy(image), steering)
                image, steering = translate_image(np.copy(image), steering)
                image = cut_image(image)

                image = (image - 128.0) / 128.0

                X_batch = np.vstack([X_batch, np.reshape(image, [1, 288, 72, 3])])
                y_batch = np.vstack([y_batch, steering])

            # Get rid of the first empty row
            X_batch = X_batch[1:, :, :, :]
            y_batch = y_batch[1:]

            yield (X_batch, y_batch)

#----------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

model = Sequential()

# Layer 1: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
model.add(Convolution2D(24, 5, 5, input_shape=(288, 72, 3), border_mode='same', init='normal'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2)))
#model.add(BatchNormalization())

# Layer 2: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
model.add(Convolution2D(36, 5, 5, init='normal'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2)))
#model.add(BatchNormalization())

# Layer 3: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
model.add(Convolution2D(48, 5, 5, init='normal'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2)))
#model.add(BatchNormalization())

# Layer 4: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
model.add(Convolution2D(64, 3, 3, init='normal'))
model.add(Activation('elu'))
#model.add(BatchNormalization())

# Layer 5: Input: 72 x 288 x 3 - Output: 35 x 143 x 24
model.add(Convolution2D(64, 3, 3, init='normal'))
model.add(Activation('elu'))
#model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(100, init='normal'))
model.add(Dropout(p=0.5))
#model.add(BatchNormalization())
model.add(Dense(50, init='normal'))
model.add(Dropout(p=0.5))
#model.add(BatchNormalization())
model.add(Dense(10, init='normal'))
model.add(Dropout(p=0.5))
#model.add(BatchNormalization())
model.add(Dense(1, init='normal'))

#---------------------------------------------------------------
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# compile and fit model
print("Fitting model")
model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=0.001))

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=2, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='model.weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
learning_rate_plateau_reducer = ReduceLROnPlateau(verbose=1, patience=0, epsilon=1e-5)
batch_size=40

fit = model.fit_generator(myGenerator(X_train, y_train, batch_size),
                          verbose=1, samples_per_epoch=len(X_train),
                          nb_epoch=10,
                          callbacks=[model_checkpoint, learning_rate_plateau_reducer, early_stopping],
                          validation_data=myGenerator(X_valid, y_valid, batch_size, "validation"),
                          nb_val_samples = len(X_valid))


# compare model predicted steering angles with labeled values
y_train_predict = model.predict(X_train)
print(y_train_predict.shape)

np.set_printoptions(suppress=True)
print(y_train_predict[0:40].T)
print(y_train[0:40].T)

# output model
print("Saving model structure and weights")
model_json = model.to_json()
import json
with open ('model.json', 'w') as f:
    json.dump(model_json, f, indent=4, sort_keys=True, separators=(',', ':'))

model.save_weights('model.h5')