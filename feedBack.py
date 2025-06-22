import cv2
import numpy as np
import threading
import time
import socket
from pyapriltags import Detector
import matplotlib.pyplot as plt
import math

TCP_IP = "127.0.0.1"
TCP_PORT = 5015
videopath = '/Users/laptopjp/Desktop/SourceDATn/birdseye_view_only.mp4'

REFERENCE_POINTS = {
    2: (-2.2, 1.3),
    3: (2.2, 1.3),
    4: (2.2, -1.3),
    5: (-2.2, -1.3)
}

detector = Detector(families="tag25h9")
H = None  # Homography matrix

# Shared feedback variables
car_x = car_y = car_yaw = None
x_car = y_car = None

feedback_ready = threading.Event()
feedback_lock = threading.Lock()

def get_homography_matrix():
    global H
    src_pts, dst_pts = [], []
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
        print("Không phát hiện đủ tag góc để tính H!")

def pixel_to_real(pixel_point):
    if H is None:
        return None
    pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1])
    real_homogeneous = np.dot(H, pixel_homogeneous)
    car_x = (real_homogeneous[0] / real_homogeneous[2]) * 0.96
    car_y = (real_homogeneous[1] / real_homogeneous[2]) * 0.96
    return (car_x, car_y)

def detect_feedback_position():
    global car_x, car_y, car_yaw, car_yaw_rad, x_car, y_car
    cap = cv2.VideoCapture(0)
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
                    with feedback_lock:
                        car_x, car_y = real_coords
                        top_left, top_right = tag.corners[0], tag.corners[1]
                        mid_x = (top_left[0] + top_right[0]) / 2
                        mid_y = (top_left[1] + top_right[1]) / 2
                        car_dir = np.array([mid_x - tag_x, mid_y - tag_y])
                        ref_dir = np.array(REFERENCE_POINTS[2]) - np.array(REFERENCE_POINTS[3])
                        car_yaw_rad = np.arctan2(
                            car_dir[1] * ref_dir[0] - car_dir[0] * ref_dir[1],
                            car_dir[0] * ref_dir[0] + car_dir[1] * ref_dir[1]
                        )
                        car_yaw_rad = -car_yaw_rad  # Invert yaw to match the coordinate system
                        car_yaw = np.rad2deg(car_yaw_rad)
                        x_car = (2.2 - car_x) * (440 / 4.4)
                        y_car = (1.3 - car_y) * (260 / 2.6)
                        print(f"[FEEDBACK] X: {x_car:.2f}, Y: {y_car:.2f}, Yaw: {car_yaw_rad:.2f}")
                        feedback_ready.set()
        time.sleep(0.1)
    cap.release()

def send_feedback():
    global x_car, y_car, car_yaw_rad
    while True:  # Keep trying to connect
        try:
            # Tạo kết nối đến server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((TCP_IP, TCP_PORT))
            print(f"Đã kết nối đến server tại {TCP_IP}:{TCP_PORT}")

            while True:
                feedback_ready.wait()  # Wait for feedback to be ready
                with feedback_lock:
                    if x_car is not None and y_car is not None and car_yaw_rad is not None:
                        # car_yaw = math.radians(car_yaw)
                        message = f"{x_car},{y_car},{car_yaw_rad}\n"
                        sock.sendall(message.encode("utf-8"))
                        print(f"Đã gửi feedback: {message.strip()}")
                        feedback_ready.clear()
                time.sleep(0.1)

        except ConnectionRefusedError:
            print("Không thể kết nối đến server. Đang thử lại sau 1 giây...")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"Lỗi khi kết nối hoặc gửi dữ liệu: {e}")
            time.sleep(1)
            continue
        finally:
            try:
                sock.close()
                print("Đã đóng kết nối.")
            except:
                pass

def generate_map_lines_and_curves():
    lines, curves = [], []

    lines.append(((116, 0), (324, 0)))
    lines.append(((116, 260), (324, 260)))
    lines.append(((0, 116), (0, 144)))
    lines.append(((440, 116), (440, 144)))

    lines += [
        ((116, 30), (192, 30)), ((248, 30), (324, 30)),
        ((116, 230), (192, 230)), ((248, 230), (324, 230)),
        ((192, 30), (192, 102)), ((248, 30), (248, 102)),
        ((192, 158), (192, 230)), ((248, 158), (248, 230)),
        ((32, 102), (100, 102)), ((100, 102), (100, 72)), ((100, 72), (160, 72)),
        ((160, 72), (160, 102)), ((160, 102), (192, 102)), ((248, 102), (271, 102)),
        ((271, 102), (271, 46)), ((271, 46), (303, 46)), ((303, 46), (303, 102)),
        ((303, 102), (408, 102)), ((408, 158), (248, 158)), 
        ((192, 158), (119, 158)), ((119, 158), (154, 193)),
        ((154, 193), (128, 214)), ((128, 214), (72, 158)), ((72, 158), (32, 158))
    ]

    curves += [
        ((116, 116), 116, 180, 270), ((324, 116), 116, 270, 360),
        ((116, 144), 116, 90, 180), ((324, 144), 116, 0, 90),
        ((116, 116), 86, 190, 270), ((324, 116), 86, 270, 350),
        ((116, 144), 86, 90, 170), ((324, 144), 86, 10, 90)
    ]

    return lines, curves

def plot_car_position_and_yaw(x, y, yaw):
    plt.plot(x, y, 'ro', markersize=8)
    # Normalize yaw to [0, 360)
    yaw = yaw % 360
    dx = 15 * math.cos(math.radians(-yaw))
    dy = 15 * math.sin(math.radians(-yaw))
    plt.arrow(x, y, dx, dy, head_width=5, head_length=8, fc='r', ec='r')

def plot_map_and_car(x_car, y_car, car_yaw):
    lines, curves = generate_map_lines_and_curves()
    plt.clf()
    # Draw lines
    for start, end in lines:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k', linewidth=1.5)
    # Draw curves
    for (cx, cy), r, a1, a2 in curves:
        angles = np.radians(np.arange(a1, a2 + 1))
        x_vals = cx + r * np.cos(angles)
        y_vals = cy + r * np.sin(angles)
        plt.plot(x_vals, y_vals, 'k', linewidth=1.5)
    # Draw car
    if x_car is not None and y_car is not None and car_yaw is not None:
        plot_car_position_and_yaw(x_car, y_car, car_yaw)
    plt.title("Car Position and Yaw on Map")
    plt.axis("equal")
    plt.xlim(0, 460)
    plt.ylim(0, 270)
    plt.pause(0.01)


if __name__ == "__main__":
    get_homography_matrix()
    if H is None:
        print("Không tìm được ma trận homography!")
    else:
        try:
            plt.ion()  # Enable interactive mode for live updating
            t1 = threading.Thread(target=detect_feedback_position, daemon=True)
            t2 = threading.Thread(target=send_feedback, daemon=True)
            t1.start()
            t2.start()
            # Plotting loop in main thread
            while True:
                with feedback_lock:
                    plot_map_and_car(x_car, y_car, car_yaw)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Dừng chương trình.")
