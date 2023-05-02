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
        print(hsv_image[320, 240])
        lower = (hsv_image[320, 240][0] - 15, hsv_image[320, 240][1] - 70, hsv_image[320, 240][2] - 40)
        upper = (hsv_image[320, 240][0] + 15, hsv_image[320, 240][1] + 70, hsv_image[320, 240][2] + 40)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        print(lower)
        print(upper)
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
