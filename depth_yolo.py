import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
import time


# Khởi tạo mô hình YOLO
model = YOLO("depthcam-use-hsv-detection/depthcam/yolo11n.engine", task="detect")


# Cấu hình Intel RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

# Cấu hình
ALPHA = 0.01  # Hệ số lọc nhiễu (càng cao càng phản hồi nhanh)
MAX_DEPTH = 6000  
depth_history = deque(maxlen=20)

prev_time = time.time()

def get_distance(depth_frame, x, y):
    """Lấy khoảng cách từ Depth Camera với bộ lọc mượt."""
    depth = depth_frame.get_distance(x, y) * 1000  
    if depth == 0 or depth > MAX_DEPTH:
        return depth_history[-1] if depth_history else None
    prev_depth = depth_history[-1] if depth_history else depth
    smoothed_depth = ALPHA * depth + (1 - ALPHA) * prev_depth
    depth_history.append(smoothed_depth)
    return round(smoothed_depth)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    current_time = time.time()
    delta = current_time - prev_time
    fps = 1.0 / delta if delta > 0 else 0
    prev_time = current_time

    if not depth_frame or not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    detections = np.empty((0, 6))
    
    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False)
    
    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            
            if conf > 60:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Tính tọa độ trung tâm bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = get_distance(depth_frame, cx, cy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                cv2.putText(frame, f'Dist: {distance}mm', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                new_detections = np.array([x1, y1, x2, y2, conf, classindex])
                detections = np.vstack((detections, new_detections))

    fps_text = f'FPS: {int(fps)}'
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow('Object Detection', frame)                        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()