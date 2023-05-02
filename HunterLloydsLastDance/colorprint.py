import pyrealsense2 as rs
import numpy as np
import cv2

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
colorDict = {"pink":((160,130,171),(175,230,255)),
             "yellow":((30,150,140),(40,240,255)),
             "orange":((10,130,200),(30,220,255)),
             "blue":((85,200,140),(100,256,255)),
             "green":((50,120,160),(70,210,255)),}


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        mask = cv2.inRange(hsv_image, np.asarray((0,0,0)),np.asarray((127,127,127)))
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        color_colormap_dim = color_image.shape
        color = "Colors: "
        for key,val in colorDict.items():
            if(hsv_image[240, 320][0]>val[0][0] and
            hsv_image[240, 320][0]<val[1][0] and
            hsv_image[240, 320][1]>val[0][1] and
            hsv_image[240, 320][1]<val[1][1] and
            hsv_image[240, 320][2]>val[0][2] and
            hsv_image[240, 320][2]<val[1][2]):
                color = color+key+" "
                mask = cv2.inRange(hsv_image, np.asarray(colorDict[key][0]),np.asarray(colorDict[key][1]))


        print(color)
        print(hsv_image[240, 320])
        cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("window", mask)


        # Show images
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


finally:

    # Stop streaming
    pipeline.stop()
