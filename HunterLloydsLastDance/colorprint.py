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
colorDict = {"pink":((154,20,111),(184,255,255)),
             "yellow":((21,20,88),(42,80,255)),
             "orange":((0,80,108),(20,255,255)),
             "blue":((82,70,88),(112,255,255)),
             "green":((42,20,67),(70,255,200)),}

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.normalize(color_image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        color_colormap_dim = color_image.shape
        lower = (hsv_image[320, 240][0] - 15, hsv_image[320, 240][1] - 70, hsv_image[320, 240][2] - 40)
        upper = (hsv_image[320, 240][0] + 15, hsv_image[320, 240][1] + 70, hsv_image[320, 240][2] + 40)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        color = "Colors: "
        for key,val in colorDict.items():
            if(hsv_image[320, 240][0]>val[0][0] and
            hsv_image[320, 240][0]<val[1][0] and
            hsv_image[320, 240][1]>val[0][1] and
            hsv_image[320, 240][1]<val[1][1] and
            hsv_image[320, 240][2]>val[0][2] and
            hsv_image[320, 240][2]<val[1][2]):
                color = color+key+" "

        print(color)
        print(hsv_image[320, 240],lower,upper)
        mask = cv2.inRange(hsv_image, lower, upper)
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
