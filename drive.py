import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import tool


sio = socketio.Server()
app = Flask(__name__)


model = None
prev_image_array = None


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
set_speed = 18
controller.set_desired(set_speed)

#registering event handler for the server
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

        correcting_factor = 1.5  # Got from experiment
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = tool.preprocess(image)  # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))*correcting_factor  #add correcting_factor
            # if steering angle is big, lower the desired speed
            # speed range: [set_speed - 8, set_speed]
            global set_speed
            controller.set_desired(set_speed - 3*(abs(steering_angle)))
            # tell the model how hard it should push the throttle to reach the desired speed
            throttle = controller.update(float(speed))
            # if no steering need and the speed reached 25, throttle = 0.35 to keep the speed
            # throttle = max(throttle, -0.15 / 0.05 * abs(steering_angle) + 0.35)

            reverse = max((speed - controller.set_point)/4-0.5, 0)
            if reverse > 0:
                throttle = 0
            else:
                throttle = max(throttle, -0.15 / 0.05 * abs(steering_angle) + 0.35)

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
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    model = load_model(args.model)

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

    app = socketio.Middleware(sio, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
