import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
import cv2

def cut_image(img):
    return img[60:136, :, :]

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

import collections
steering_vector = collections.deque(maxlen=5)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    # Size of the image
    model_height = 64
    model_weight = 64

    image_cuted = cut_image(np.copy(image))

    image_resized = cv2.resize(image_cuted,(model_height, model_weight), interpolation = cv2.INTER_AREA)
    image_array = np.reshape(image_resized, [1, model_height, model_weight, 3])

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(image_array, batch_size=1))
    steering_vector.append(steering_angle)

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle_desired = 0.3 - abs(steering_angle) * 0.05
    if speed < throttle_desired*100*0.8:
        throttle = 1
    elif speed > throttle_desired*100*1.2:
        throttle = -1
    else:
        throttle = throttle_desired

    print("\tSteering: {:+2.4f}  Speed: {:2.2f}".format(steering_angle, throttle_desired*100))
    send_control(np.median(steering_vector), throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        # model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)