# parsing command line arguments
import argparse
# decoding camera images
import base64
# for frametimestamp saving
from datetime import datetime
# reading and writing files
import os
# high level file operations
import shutil
import numpy as np
# real-time server
import socketio
# concurrent networking
import eventlet.wsgi
# image manipulation
from PIL import Image
# web framework
from flask import Flask
# input output
from io import BytesIO
# load our saved model
import tensorflow as tf
# helper class
import tf_model

import model_structure as model

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)

# init our model and image array as empty
prev_image_array = None

sess = tf.InteractiveSession()
saver = tf.train.Saver()

class SimplePIController:
    def __init__(self, coef_c, coef_t):
        # initiation
        self.coef_c = coef_c
        self.coef_t = coef_t
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, current_speed):
        # immediate error
        self.error = self.set_point - current_speed

        # total error
        self.integral += self.error

        return self.coef_c * self.error + self.coef_t * self.integral


controller = SimplePIController(0.08, 0.001)
set_speed = 25
controller.set_desired(set_speed)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        # steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        # throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        correcting_factor = 1.4  # Got from experiment
        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = tf_tool.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = model.y_.eval(feed_dict={model.X: image, model.keep_prob: 1.0})[0][0]
            # if steering angle is big, lower the desired speed
            # speed range: [set_speed - 8, set_speed]
            global set_speed
            controller.set_desired(set_speed - 8 * (steering_angle ** 2))
            # tell the model how hard it should push the throttle to reach the desired speed
            throttle = controller.update(float(speed))
            # if no steering need and the speed reached 25, throttle = 0.35 to keep the speed
            throttle = max(throttle, -0.15 / 0.05 * abs(steering_angle) + 0.35)

            reverse = max((speed - set_speed) / 5, 0)

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle, reverse)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0)


def send_control(steering_angle, throttle, reverse):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'reverse': reverse.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    # parser.add_argument(
    #     'model',
    #     type=str,
    #     help='Path to model h5 file. Model should be on the same path.'
    # )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # load model
    with tf.Session() as sess:
        saver.restore(sess, "saved_models/model-1.ckpt")

        if args.image_folder != '':
            print("Creating image folder at {}".format(args.image_folder))
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            print("RECORDING THIS RUN ...")
        else:
            print("NOT RECORDING THIS RUN ...")

        # wrap Flask application with engineio's middleware
        app = socketio.Middleware(sio, app)

        # deploy as an eventlet WSGI server
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
