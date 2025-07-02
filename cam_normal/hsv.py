import cv2
import numpy as np
import time

def detect_circles(frame, contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * radius * radius
            ratio = area / circle_area
            if 0.7 < ratio < 1.2:
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                cv2.putText(frame, "Circle", (int(x)-20, int(y)-radius-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

def detect_red_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mã màu đỏ
    lower_red1 = np.array([180, 140, 50])
    upper_red1 = np.array([180, 255, 255])
    lower_red2 = np.array([170, 140, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm contour có diện tích lớn nhất
    best_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 700 and area > max_area:
            best_cnt = cnt
            max_area = area

    # Nếu có contour tốt nhất thì vẽ
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setting_bright(frame) :
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Tính độ sáng trung bình từ kênh V
    brightness = int(np.mean(v))
    cv2.putText(frame, f"Brightness: {brightness}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# Mở camera
cap = cv2.VideoCapture(0)

# Biến đếm thời gian để tính FPS
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tính FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Resize
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 70)

    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nhận diện
    detect_circles(frame, contours)
    detect_red_objects(frame)
    setting_bright(frame)

    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow("Frame", frame)
    cv2.imshow("Edge", eroded)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
