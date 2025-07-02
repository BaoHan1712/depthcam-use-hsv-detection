import cv2
import pyrealsense2 as rs
import numpy as np
import time
from cover.utils import *


# === Thiết lập màu đỏ ===
lower_red1 = np.array([15, 116, 144])
upper_red1 = np.array([84, 255, 255])
lower_red2 = np.array([170, 116, 144])
upper_red2 = np.array([180, 255, 255])

# Khởi tạo RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

ALPHA = 0.1  
depth_history = {}

prev_time = time.time()

def get_smoothed_distance(depth_frame, x, y, key, history_size=20):
    """ Hàm lọc nhiễu khoảng cách từ depth cam """
    depth = depth_frame.get_distance(x, y) * 1000  # mm
    if depth == 0 or depth > 5000:
        return None
    if key not in depth_history:
        depth_history[key] = []
    prev_depth = depth_history[key][-1] if depth_history[key] else depth
    smoothed = ALPHA * depth + (1 - ALPHA) * prev_depth
    depth_history[key].append(smoothed)
    if len(depth_history[key]) > history_size:
        depth_history[key].pop(0)
    return round(smoothed)

def detect_red_object(frame):
    """ Hàm phát hiện vật thể đỏ """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800 and area > max_area:
            max_area = area
            best_cnt = cnt

    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        cx, cy = x + w // 2, y + h // 2
        return (x, y, w, h), (cx, cy)
    return None, (-1, -1)


def get_red_object_center(frame):
    """ Trả về tọa độ (cx, cy) của tâm vật thể đỏ nếu phát hiện được """
    _, (cx, cy) = detect_red_object(frame)
    cv2.putText(frame, f" ({cx}, {cy})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return (cx, cy)

while True:
    current_time = time.time()
    delta = current_time - prev_time
    fps = 1.0 / delta if delta > 0 else 0
    prev_time = current_time

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    bbox, (cx, cy) = detect_red_object(frame)
    zone = get_object_position_zone(cx, frame)
    if zone != "none":
        cv2.putText(frame, f"Zone: {zone.upper()}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if cx != -1 and cy != -1:
        distance = get_smoothed_distance(depth_frame, cx, cy, "red_object")
        if distance:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Dist: {distance} mm", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    get_red_object_center(frame)
    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

   


    cv2.imshow("Red Object Depth", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()


########### ĐANG THIẾU XỬ LÝ BIÊN ######### ########