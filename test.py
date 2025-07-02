import cv2
import numpy as np
import time

# === Thiết lập mã màu ===
COLOR_MODE_1 = [
    np.array([180, 140, 50]), np.array([180, 255, 255]),
    np.array([170, 140, 50]), np.array([180, 255, 255])
]

COLOR_MODE_2 = [
    np.array([0, 100, 100]), np.array([40, 255, 255]),
    np.array([170, 100, 100]), np.array([180, 255, 255])
]

# Trạng thái
current_mode = 1
prev_brightness = -1
no_red_counter = 0
mode2_start_time = None  # Thời điểm bắt đầu mode 2


def get_object_position_zone(cx, frame):
    frame_width = frame.shape[1]

    # Chia vùng: left 40%, center 20%, right 40%
    left_boundary = int(frame_width * 0.4)
    right_boundary = int(frame_width * 0.6)

    # Vẽ đường chia vùng
    cv2.line(frame, (left_boundary, 0), (left_boundary, frame.shape[0]), (100, 100, 255), 2)
    cv2.line(frame, (right_boundary, 0), (right_boundary, frame.shape[0]), (100, 100, 255), 2)

    # Vẽ nhãn vùng
    cv2.putText(frame, "LEFT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, "CENTER", (left_boundary + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "RIGHT", (right_boundary + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Xác định vị trí vật thể
    if cx == -1:
        return "none"
    elif cx < left_boundary:
        return "left"
    elif cx < right_boundary:
        return "center"
    else:
        return "right"


def detect_red_objects(frame, hsv):
    global current_mode, no_red_counter

    if current_mode == 1:
        lower_red1, upper_red1, lower_red2, upper_red2 = COLOR_MODE_1
    else:
        lower_red1, upper_red1, lower_red2, upper_red2 = COLOR_MODE_2

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_red = False
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000 and area > max_area:
            max_area = area
            best_cnt = cnt

    cx, cy = -1, -1  # Mặc định -1 nếu không có vật thể

    if best_cnt is not None:
        found_red = True
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        M = cv2.moments(best_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if found_red:
        no_red_counter = 0
    else:
        no_red_counter += 1

    return no_red_counter >= 5, (cx, cy)


def detect_circles(frame, contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * radius * radius
            ratio = area / circle_area
            if 0.7 < ratio < 1.2:
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                cv2.putText(frame, "Circle", (int(x) - 20, int(y) - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def setting_bright(frame, hsv):
    global prev_brightness

    v = hsv[:, :, 2]
    brightness = int(np.mean(v))
    cv2.putText(frame, f"Brightness: {brightness}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if prev_brightness != -1:
        delta = abs(brightness - prev_brightness)
        if delta > 10:
            return True
    prev_brightness = brightness
    return False


# === MAIN ===
cap = cv2.VideoCapture(0)
prev_time = time.time()
mode2_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness_changed = setting_bright(frame, hsv)
    red_not_found, (cx, cy) = detect_red_objects(frame, hsv)

    # Nếu sáng thay đổi và không thấy vật thể đỏ → chuyển sang MODE 2
    if brightness_changed and red_not_found and current_mode == 1:
        current_mode = 2
        mode2_start_time = time.time()
        print("[!] Brightness changed + no red → Switched to COLOR_MODE_2")

    # Nếu đang ở MODE 2 quá 8 giây → quay về MODE 1
    if current_mode == 2 and mode2_start_time is not None:
        if time.time() - mode2_start_time > 8:
            current_mode = 1
            mode2_start_time = None
            print("[✔] Reverted to COLOR_MODE_1 after 8 seconds")

    # Phân vùng trái - giữa - phải
    zone = get_object_position_zone(cx, frame)
    if zone != "none":
        cv2.putText(frame, f"Zone: {zone.upper()}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)


    # Phát hiện hình tròn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 70)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detect_circles(frame, contours)

    # Hiển thị thông tin
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"MODE: {current_mode}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Edge", eroded)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
