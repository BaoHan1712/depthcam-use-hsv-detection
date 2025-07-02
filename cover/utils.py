import cv2


def get_object_position_zone(cx, frame):
    """ hàm lấy vị trí vật đang ở bên nào"""
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
    
