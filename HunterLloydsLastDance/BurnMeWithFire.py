import time

import pyrealsense2

import pyrealsense2 as rs
import numpy as np
import cv2

from maestro import Controller

tango = Controller()
tango.setAccel(0, 0)
tango.setAccel(1, 0)


def forward():
    tango.setTarget(1, 6000)
    tango.setTarget(0, 5000)


def back():
    tango.setTarget(1, 6000)
    tango.setTarget(0, 7000)


def left():
    tango.setTarget(0, 6000)
    tango.setTarget(1, 5000)


def right():
    tango.setTarget(0, 6000)
    tango.setTarget(1, 7000)


def stop():
    tango.setTarget(0, 6000)
    tango.setTarget(1, 6000)

def tickLeft():
    left()
    time.sleep(0.3)
    stop()
    time.sleep(0.1)

class StateMachine:
    INITIAL_FIND = 0
    PRE_FIELD = 1
    FIELD_HUNTING = 2
    COLOR_ID = 3
    RETURN_HUNTING = 4
    RETURN_PRE_FIELD = 5
    COLOR_HUNTER = 6
    DONE = 7

    def __init__(self):
        self.state = self.INITIAL_FIND

    def initial_find(self, frame, tango, window):
        print("Finding")
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, frame)

    def pre_field(self, frame, tango, window):
        print("Going To Field")

    def field_hunting(self, frame, tango, window):
        print("In Field")

    def color_id(self, frame, tango, window):
        print("Show Me The Color")

    def return_hunting(self, frame, tango, window):
        print("Finding")

    def return_pre_field(self, frame, tango, window):
        print("Going To Field")

    def color_hunter(self, frame, tango, window): \
            print("Finding Color")

    def done(self, frame, tango, window):
        print("Fini")

    def tick(self, frame, tango, window):
        if (self.state == self.INITIAL_FIND):
            self.initial_find(frame, tango, window)
        elif (self.state == self.PRE_FIELD):
            self.pre_field(frame, tango, window)
        elif (self.state == self.FIELD_HUNTING):
            self.field_hunting(frame, tango, window)
        elif (self.state == self.COLOR_ID):
            self.color_id(frame, tango, window)
        elif (self.state == self.RETURN_HUNTING):
            self.return_hunting(frame, tango, window)
        elif (self.state == self.RETURN_PRE_FIELD):
            self.return_pre_field(frame, tango, window)
        elif (self.state == self.COLOR_HUNTER):
            self.color_hunter(frame, tango, window)
        else:
            self.done(frame, tango, window)

robot = StateMachine()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        color_colormap_dim = color_image.shape




        # Show images
        robot.tick(color_image,tango,"main")
        key = cv2.waitKey(1)
        tickLeft()

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


finally:

    # Stop streaming
    pipeline.stop()
    tango.setTarget(0, 6000)