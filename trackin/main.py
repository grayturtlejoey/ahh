## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
import pyrealsense2
###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from maestro import Controller

tango = Controller()

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

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
tracker = cv2.TrackerKCF_create()

bbox = cv2.selectROI("select",color_image)
cv2.destroyWindow("select")

ok = tracker.init(color_image, bbox)

targetDepth = -1

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        timer = cv2.getTickCount()
        ok, bbox = tracker.update(color_image)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(color_image, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(color_image, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)



        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        map = np.zeros((depth_colormap_dim[0], 2 * depth_colormap_dim[1], depth_colormap_dim[2]), dtype=np.uint8)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))


        cv2.line(map, (0,0),(640,320), (255,255,255),2)
        cv2.line(map, (1280, 0), (640, 320), (255, 255, 255), 2)
        cv2.rectangle(map, (635, 318), (645, 322), (0, 0, 255), 5, 1)
        sectionDepth = depth_image[p1[0]:p2[0],p1[1]:p2[1]]
        try:
            rawDepth = np.nanmedian(sectionDepth)
            boxDepthAvg = 0.001*rawDepth
            if(targetDepth == -1):
                targetDepth = rawDepth
        except:
            boxDepthAvg = 0.0000001
        
        print(f"{rawDepth},{targetDepth}")
        if(targetDepth-rawDepth>70):
            tango.setTarget(0,6800)
            print("forward")
        elif(targetDepth-rawDepth<-70):
            tango.setTarget(0,5200)
            print("backward")
        else:
            tango.setTarget(0,6000)



        try:
            cv2.rectangle(map,
                          (int(((p1[0]*2-640)/1280)*int(1280*boxDepthAvg*0.66)+640)-int((p2[0]-p1[0])/2),318-int(boxDepthAvg*240)),
                          (int(((p2[0]*2-640)/1280)*int(1280*boxDepthAvg*0.66)+640)+int((p2[0]-p1[0])/2),322-int(boxDepthAvg*240)),
                          (255, 0, 0),
                          2,
                          1)
        except:
            print("Depth Data Issue")
        img2 = np.vstack((images, map))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', img2)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()
    tango.setTarget(0,6000)
