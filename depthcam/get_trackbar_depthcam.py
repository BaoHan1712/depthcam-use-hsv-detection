import cv2
import numpy as np
import pyrealsense2 as rs

def nothing(x):
    pass

# Tạo cửa sổ trackbar
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 300)

cv2.createTrackbar("Hue Min", "Trackbars", 20, 180, nothing)
cv2.createTrackbar("Hue Max", "Trackbars", 30, 180, nothing)
cv2.createTrackbar("Sat Min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Val Min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, nothing)


# Khởi tạo RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8,30)
profile = pipeline.start(config)

# Căn chỉnh ảnh màu và depth
align = rs.align(rs.stream.color)

print("Nhấn 'o' để xuất mã màu HSV hiện tại.")
print("Nhấn 'q' để thoát.")

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()

    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tính độ sáng
    brightness = int(np.mean(v))

    # Lấy giá trị từ trackbar
    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    # Tạo mask theo 2 khoảng đỏ (Hue thấp và Hue cao)
    lower_red1 = np.array([h_min, s_min, v_min])
    upper_red1 = np.array([h_max, s_max, v_max])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([180, s_max, v_max])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Hiển thị độ sáng
    cv2.putText(frame, f"Brightness: {brightness}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Hiển thị
    cv2.imshow("Original", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Red Detected", red_result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o'):
        print("\n[✔] Mã màu bạn chọn:")
        print(f"lower_red1 = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_red1 = np.array([{h_max}, {s_max}, {v_max}])")
        print(f"lower_red2 = np.array([170, {s_min}, {v_min}])")
        print(f"upper_red2 = np.array([180, {s_max}, {v_max}])")

pipeline.stop()
cv2.destroyAllWindows()
