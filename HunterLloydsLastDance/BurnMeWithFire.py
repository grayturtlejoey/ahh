import time

import pyrealsense2

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

from maestro import Controller

tango = Controller()
tango.setAccel(1, 0)
tango.setAccel(0, 0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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


def zero():
    tango.setTarget(0, 6000)
    tango.setTarget(1, 6000)
    tango.setTarget(2,6000)
    tango.setTarget(3,6000)
    tango.setTarget(4,5500)
def lookForward():
    tango.setTarget(4,5500)

def lookDown():
    tango.setTarget(4,2000)

def tickLeft():
    left()
    time.sleep(0.2)
    stop()
    time.sleep(0.5)

def tickRight():
    right()
    time.sleep(0.2)
    stop()
    time.sleep(0.5)

def tickForward():
    forward()
    time.sleep(0.5)
    stop()
    time.sleep(0.5)

class StateMachine:
    INITIAL_FIND = 0
    PRE_FIELD = 1
    FIELD_HUNTING = 2
    COLOR_ID = 3
    RETURN_HUNTING = 4
    RETURN_PRE_FIELD = 5
    COLOR_HUNTER = 6
    DONE = 7
    BLUE = [(0,0,0),(255,255,255)]

    def __init__(self):
        self.state = self.COLOR_ID
        self.markerX = -1
        self.markerY = -1
        self.falseAlarm = 0


    def initial_find(self, frame, tango, window):
        print("Finding")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,parameters=parameters,)
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                if(markerID==49):
                    self.markerX = cX
                    self.markerY = cY
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(frame, str(markerID),
                            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)


        # Display the resulting frame
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, frame)

        if(self.markerX>0):
            if(self.markerX<300):
                tickRight()
            elif(self.markerX>340):
                tickLeft()
            else:
                stop()
                self.state = self.PRE_FIELD
        else:
            tickLeft()

    def pre_field(self, frame, tango, window):
        print("Going To Field")
        lookDown()
        forward()
        time.sleep(1.2)
        stop()
        lookForward()
        self.state = self.FIELD_HUNTING
        self.markerX = -1
        self.markerY = -1

    def field_hunting(self, frame, tango, window):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
        # Draw the rectangle around each face
        self.markerX = -1
        self.markerY = -1
        wi = 0
        he = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.markerX = int(x+w/2)
            cv2.line(frame, (self.markerX,0), (self.markerX,480), (0, 255, 0), 2)
            wi = w
            he = h
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, frame)


        if(wi*he<35000):
            print(wi*he)
            if (self.markerX > 0):
                if (self.markerX < 300):
                    tickRight()
                elif (self.markerX > 340):
                    tickLeft()
                else:
                    tickForward()
            else:
                tickLeft()
        else:
            self.falseAlarm += 1
            if self.falseAlarm>10:
                self.state = self.COLOR_ID


        print("In Field")


    def color_id(self, frame, tango, window):


        average_color_row = np.average(frame[220:260,300:340], axis=0)
        average_color = np.average(average_color_row, axis=0)
        average_color = [int(average_color[0]),int(average_color[1]),int(average_color[2])]
        print(average_color)
        cv2.rectangle(frame, (300, 220), (340, 260), average_color, 2)
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, frame)

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
stop()
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
lookForward()
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

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


finally:

    # Stop streaming
    pipeline.stop()
    tango.setTarget(0, 6000)