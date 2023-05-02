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

colorDict = {"pink":((160,130,171),(175,230,256)),
             "yellow":((30,150,140),(40,240,256)),
             "orange":((10,130,200),(30,220,256)),
             "blue":((85,200,140),(100,256,256)),
             "green":((50,120,160),(75,210,256)),}

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

def lookDownish():
    tango.setTarget(4,3300)

def tickLeft():
    left()
    time.sleep(0.2)
    stop()
    time.sleep(0.5)

def tickLeftFast():
    left()
    time.sleep(0.4)
    stop()
    time.sleep(0.2)
def tickRight():
    right()
    time.sleep(0.2)
    stop()
    time.sleep(0.5)

def tickRightFast():
    right()
    time.sleep(0.4)
    stop()
    time.sleep(0.2)

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
        self.state = self.COLOR_HUNTER
        self.markerX = -1
        self.markerY = -1
        self.falseAlarm = 0
        self.colorName = "None"
        self.newTime = time.time()
        self.timer = 5



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
            tickLeftFast()

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
                self.newTime = time.time()
                self.timer = 5


        print("In Field")


    def color_id(self, frame, tango, window):
        if (self.timer <= 0):
            frame = cv2.putText(frame, self.colorName +" has been selected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window, frame)
            time.sleep(2)
            self.state = self.RETURN_HUNTING
            self.markerX = -1
            self.markerY = -1
            return()


        self.oldTime = self.newTime
        self.newTime = time.time()
        deltatime = self.newTime-self.oldTime

        average_color_row = np.average(frame[220:260,300:340], axis=0)
        average_color = np.average(average_color_row, axis=0)
        average_color = [int(average_color[0]),int(average_color[1]),int(average_color[2])]
        cv2.rectangle(frame, (300, 220), (340, 260), average_color, -1)
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        print(average_color,hsv_image[240,320])


        for key,val in colorDict.items():
            if(hsv_image[240, 320][0]>val[0][0] and
            hsv_image[240, 320][0]<val[1][0] and
            hsv_image[240, 320][1]>val[0][1] and
            hsv_image[240, 320][1]<val[1][1] and
            hsv_image[240, 320][2]>val[0][2] and
            hsv_image[240, 320][2]<val[1][2]):
                if(self.colorName != key):
                    self.colorName = key
                    self.timer = 5
                self.timer -= deltatime
                color = key
                print(color)


        frame = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        frame = cv2.putText(frame, self.colorName+": "+str(round(self.timer,2)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,0), 2, cv2.LINE_AA)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, frame)



        print("Show Me The Color")


    def return_hunting(self, frame, tango, window):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, )
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
                if (markerID == 22):
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
                self.state = self.RETURN_PRE_FIELD
        else:
            tickLeftFast()
        print("Finding")

    def return_pre_field(self, frame, tango, window):
        print("Going To Field")
        lookDown()
        forward()
        print("Going To Field")
        tickForward()
        print("Going To Field")
        tickForward()
        print("Going To Field")
        stop()
        lookForward()
        self.state = self.FIELD_HUNTING
        self.markerX = -1
        self.markerY = -1
        print("Finding Color")
        tickForward()
        tickForward()
        tickForward()
        tickForward()
        lookDownish()

    def color_hunter(self, frame, tango, window):
            lookDown()
            print("Finding Color")
            self.colorName = "green"
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, colorDict[self.colorName][0], colorDict[self.colorName][1])
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            frame = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            cm = cv2.moments(mask, True)
            if cm["m00"] != 0:
                cX = int(cm['m10'] / cm['m00'])
                cY = int(cm['m01'] / cm['m00'])
            else:
                cX, cY = 0, 0
            print((cX, cY))
            cv2.rectangle(frame, (cX - 3, cY - 3), (cX + 3, cY + 3), (255, 255, 255), 5, 1)


            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window, frame)


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