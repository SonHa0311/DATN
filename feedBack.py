import cv2
import numpy as np
import threading
import time
import socket
from pyapriltags import Detector
import math
import matplotlib.pyplot as plt
from collections import deque

# Server thông tin
TCP_IP = "127.0.0.1"
TCP_PORT = 5015

REFERENCE_POINTS = {
    2: (-2.2, 1.3),
    3: (2.2, 1.3),
    4: (2.2, -1.3),
    5: (-2.2, -1.3)
}

# video_path = '/Users/laptopjp/Desktop/VS_Code/HybridAStar/tester1.mp4'
detector = Detector(families="tag25h9")
H = None  # Homography matrix

car_x_start = car_y_start = car_yaw_start = None
car_x = car_y = car_yaw = None
x_car = y_car = None

# Add trajectory storage
trajectory = deque(maxlen=1000)  # Store last 1000 positions
trajectory_data = []  # Store complete trajectory data for PurePursuit
trajectory_lock = threading.Lock()

start_position_found = threading.Event()
feedback_ready = threading.Event()
feedback_lock = threading.Lock()

def get_trajectory_data():
    """Return the current trajectory data for PurePursuit"""
    with trajectory_lock:
        return list(trajectory_data)

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
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        H, _ = cv2.findHomography(src_pts, dst_pts)
    else:
        print("Không phát hiện đủ tag góc để tính H!")

def pixel_to_real(pixel_point):
    if H is None:
        return None
    pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1])
    real_homogeneous = np.dot(H, pixel_homogeneous)
    # car_x = (real_homogeneous[0] / real_homogeneous[2]) * (h_cam - h_xe) / h_cam
    # car_y = (real_homogeneous[1] / real_homogeneous[2]) * (h_cam - h_xe) / h_cam
    car_x = (real_homogeneous[0] / real_homogeneous[2])*0.96
    car_y = (real_homogeneous[1] / real_homogeneous[2])*0.96
    return (car_x, car_y)

def detect_start_position():
    global car_x_start, car_y_start, car_yaw_start
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            if tag.tag_id == 13:
                tag_x, tag_y = map(float, tag.center)
                real_coords = pixel_to_real((tag_x, tag_y))
                if real_coords:
                    car_x_start, car_y_start = real_coords
                    top_left, top_right = tag.corners[0], tag.corners[1]
                    mid_x = (top_left[0] + top_right[0]) / 2
                    mid_y = (top_left[1] + top_right[1]) / 2

                    car_dir = np.array([mid_x - tag_x, mid_y - tag_y])
                    ref_dir = np.array(REFERENCE_POINTS[2]) - np.array(REFERENCE_POINTS[3])
                    car_yaw_rad = np.arctan2(
                        car_dir[1] * ref_dir[0] - car_dir[0] * ref_dir[1],
                        car_dir[0] * ref_dir[0] + car_dir[1] * ref_dir[1]
                    )
                    car_yaw_start = np.rad2deg(car_yaw_rad)
                    x_car_start = (2.2 - car_x_start) * (440 / 4.4)
                    y_car_start = (1.3 - car_y_start) * (260 / 2.6)

                    print(f"[START] X: {x_car_start:.2f}, Y: {y_car_start:.2f}, Yaw: {car_yaw_start:.2f}")
                    start_position_found.set()
                    cap.release()
                    return

    cap.release()

def detect_feedback_position():
    global car_x, car_y, car_yaw, x_car, y_car
    start_position_found.wait()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            if tag.tag_id == 13:
                tag_x, tag_y = map(float, tag.center)
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
                        car_yaw = np.rad2deg(car_yaw_rad)

                        x_car = (2.2 - car_x) * (440 / 4.4)
                        y_car = (1.3 - car_y) * (260 / 2.6)

                        # Add position to trajectory
                        with trajectory_lock:
                            trajectory.append((x_car, y_car))
                            # Store complete trajectory data for PurePursuit
                            trajectory_data.append([x_car, y_car, car_yaw])

                        print(f"[FEEDBACK] X: {x_car:.2f}, Y: {y_car:.2f}, Yaw: {car_yaw}")
                        feedback_ready.set()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

    cap.release()

# Hàm để gửi feedback
def send_feedback():
    global x_car, y_car, car_yaw
    while True:  # Keep trying to connect
        try:
            # Tạo kết nối đến server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((TCP_IP, TCP_PORT))
            print(f"Đã kết nối đến server tại {TCP_IP}:{TCP_PORT}")

            while True:
                feedback_ready.wait()  # Wait for feedback to be ready
                with feedback_lock:
                    if x_car is not None and y_car is not None and car_yaw is not None:
                        car_yaw = math.radians(car_yaw)
                        message = f"{x_car},{y_car},{car_yaw}\n"
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
        # ((192, 158), (110, 158)), ((64, 158), (32, 158)),
        # ((110, 158), (145, 193)), ((145, 193), (123, 216)), ((123, 216), (64, 158))
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

def draw_car_position(x, y, yaw):
    plt.plot(x, y, 'ro', markersize=8)
    dx = 15 * math.cos(math.radians(-yaw))
    dy = 15 * math.sin(math.radians(-yaw))
    plt.arrow(x, y, dx, dy, head_width=5, head_length=8, fc='r', ec='r')

def draw_map_with_car(lines, curves):
    plt.clf()
    for start, end in lines:
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k', linewidth=1.5)
    for (cx, cy), r, a1, a2 in curves:
        angles = np.radians(np.arange(a1, a2 + 1))
        x_vals = cx + r * np.cos(angles)
        y_vals = cy + r * np.sin(angles)
        plt.plot(x_vals, y_vals, 'k', linewidth=1.5)

    with feedback_lock:
        if x_car is not None and y_car is not None:
            draw_car_position(x_car, y_car, car_yaw)
            
            # Plot trajectory
            with trajectory_lock:
                if len(trajectory) > 1:
                    traj_x, traj_y = zip(*trajectory)
                    plt.plot(traj_x, traj_y, 'b-', linewidth=1, alpha=0.5, label='Trajectory')
                    plt.legend()

    plt.title("Vị trí xe trên bản đồ")
    plt.axis("equal")
    plt.xlim(0, 460)
    plt.ylim(0, 270)
    plt.pause(0.01)

if __name__ == "__main__":
    get_homography_matrix()
    if H is None:
        print("Không tìm được ma trận homography!")
        exit()

    print("🗺️ Hiển thị bản đồ...")
    lines, curves = generate_map_lines_and_curves()

    # Khởi động các luồng xử lý
    t1 = threading.Thread(target=detect_start_position)
    t2 = threading.Thread(target=detect_feedback_position)
    t3 = threading.Thread(target=send_feedback)

    t1.start()
    t2.start()
    t3.start()

    # Vòng lặp vẽ bản đồ liên tục
    plt.ion()
    while True:
        draw_map_with_car(lines, curves)
        time.sleep(0.1)  # làm dịu CPU và cập nhật vị trí

    # Nếu cần kết thúc:
    # t1.join()
    # t2.join()
    # t3.join()

    
