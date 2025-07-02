import cv2
import numpy as np

def nothing(x):
    pass

# Tạo cửa sổ và trackbar
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 300)

cv2.createTrackbar("Hue Min", "Trackbars", 20, 180, nothing)
cv2.createTrackbar("Hue Max", "Trackbars", 30, 180, nothing)
cv2.createTrackbar("Sat Min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Val Min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, nothing)

# Mở camera
cap = cv2.VideoCapture(0)

print("Nhấn 'o' để xuất mã màu HSV hiện tại.")
print("Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tính độ sáng trung bình từ kênh V
    brightness = int(np.mean(v))

    # Lấy trackbar
    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    # Tạo mask đỏ (2 dải hue)
    lower_red1 = np.array([h_min, s_min, v_min])
    upper_red1 = np.array([h_max, s_max, v_max])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([180, s_max, v_max])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Hiển thị độ sáng trên góc trái
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

cap.release()
cv2.destroyAllWindows()
