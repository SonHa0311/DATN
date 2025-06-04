# start_position_sender.py
import cv2
import numpy as np
from pyapriltags import Detector
import socket
import time
import math
TCP_IP = "127.0.0.1"
TCP_PORT = 5013
REFERENCE_POINTS = {
    2: (-2.2, 1.3),
    3: (2.2, 1.3),
    4: (2.2, -1.3),
    5: (-2.2, -1.3)
}
detector = Detector(families="tag25h9")
H = None

def get_homography_matrix():
    global H
    src_pts = []
    dst_pts = []
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        detected_tags = {tag.tag_id: tag.center for tag in tags if tag.tag_id in REFERENCE_POINTS}
        if len(detected_tags) == len(REFERENCE_POINTS):
            for tag_id in [2, 3, 4, 5]:
                src_pts.append(detected_tags[tag_id])
                dst_pts.append(REFERENCE_POINTS[tag_id])
            break
    cap.release()
    if len(src_pts) == len(REFERENCE_POINTS):
        H, _ = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))
    else:
        print("Không đủ tag góc để tính H!")

def pixel_to_real(pixel_point):
    if H is None:
        return None
    pixel_h = np.array([pixel_point[0], pixel_point[1], 1])
    real_h = np.dot(H, pixel_h)
    return real_h[0] / real_h[2] * 0.96, real_h[1] / real_h[2] * 0.96

def detect_and_send_start_position():
    cap = cv2.VideoCapture(0)
    get_homography_matrix()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        for tag in tags:
            if tag.tag_id == 13:
                tag_x, tag_y = map(int, tag.center)
                real_coords = pixel_to_real((tag_x, tag_y))
                if real_coords:
                    car_x, car_y = real_coords
                    top_left, top_right = tag.corners[0], tag.corners[1]
                    mid_x = (top_left[0] + top_right[0]) / 2
                    mid_y = (top_left[1] + top_right[1]) / 2
                    car_dir = np.array([mid_x - tag_x, mid_y - tag_y])
                    ref_dir = np.array(REFERENCE_POINTS[2]) - np.array(REFERENCE_POINTS[3])
                    yaw = np.rad2deg(np.arctan2(
                        car_dir[1] * ref_dir[0] - car_dir[0] * ref_dir[1],
                        car_dir[0] * ref_dir[0] + car_dir[1] * ref_dir[1]
                    ))
                    yaw = math.radians(yaw)
                    x_map = (2.2 - car_x) * (440 / 4.4)
                    y_map = (1.3 - car_y) * (260 / 2.6)
                    message = f"START,{x_map:.2f},{y_map:.2f},{yaw:.2f}\n"
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect((TCP_IP, TCP_PORT))
                        sock.sendall(message.encode("utf-8"))
                        print(f"Đã gửi vị trí START: {message.strip()}")
                        sock.close()
                    except Exception as e:
                        print("Lỗi kết nối khi gửi START:", e)
                    cap.release()
                    return
    cap.release()

if __name__ == "__main__":
    detect_and_send_start_position()
