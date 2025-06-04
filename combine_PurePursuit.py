import json
import math
import socket
import numpy as np

L = 18  # Look ahead distance in meters
dt = 0.1  # Discrete time step (seconds)

# Vehicle parameters (meters)
LENGTH = 43
WIDTH = 21
BACKTOWHEEL = 15
WHEEL_LEN = 6
WHEEL_WIDTH = 2.2
TREAD = 16
WB = 25

# Define the vehicle class
class Vehicle:
    def __init__(self, x, y, yaw, vel=10, max_steering_angle_deg=30):  # Default for forward movement
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vel = vel
        self.max_steering_angle = math.radians(max_steering_angle_deg)  # Convert to radians

# Trajectory class for handling path points
class Trajectory:
    def __init__(self, traj_x, traj_y, traj_yaw):
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.traj_yaw = traj_yaw
        self.last_idx = 0

    def getPoint(self, idx):
        return [self.traj_x[idx], self.traj_y[idx], self.traj_yaw[idx]]

    def getTargetPoint(self, pos):
        target_idx = self.last_idx
        target_point = self.getPoint(target_idx)
        curr_dist = getDistance(pos, target_point)

        while curr_dist < L and target_idx < len(self.traj_x) - 1:
            target_idx += 1
            target_point = self.getPoint(target_idx)
            curr_dist = getDistance(pos, target_point)

        self.last_idx = target_idx
        return self.getPoint(target_idx)

# Helper function for calculating distance
def getDistance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

# PI Controller for controlling vehicle speed and steering angle
class PI:
    def __init__(self, kp=1, ki=0.01):
        self.kp = kp
        self.ki = ki
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.last_error = 0.0

    def control(self, error):
        self.Pterm = self.kp * error
        self.Iterm += error * dt
        output = self.Pterm + self.ki * self.Iterm
        return output

# Load trajectory from file
def load_trajectory_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        # Ensure each point has 3 elements; if missing yaw, default it to 0
        return [(point[0], point[1], point[2] if len(point) == 3 else 0) for point in data]
    except FileNotFoundError:
        print(f"Error: File {filename} not found.", end='\n')
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file {filename}.", end='\n')
        return []

# Function to send control commands via TCP
def send_control_command(sock, v, delta_rad):
    try:
        delta_deg = math.degrees(delta_rad)  # Convert rad to degrees
        message = f"{v / 100:.3f},{delta_deg:.3f}\n"  # Format the message
        sock.sendall(message.encode("utf-8"))
    except Exception as e:
        print(f"Lỗi khi gửi dữ liệu: {e}", end='\n')

# Function to receive real-time feedback (position data) from another program
def receive_feedback(sock):
    try:
        # Receive data from the connected client (feedback from sensors, like GPS, IMU, etc.)
        data = sock.recv(1024).decode('utf-8')  # Receive feedback data as string
        if data:
            feedback = data.split(",")  # Assuming the format is: "x,y,yaw"
            return float(feedback[0]), float(feedback[1]), float(feedback[2])  # return x, y, yaw
        return None, None, None
    except Exception as e:
        print(f"Lỗi khi nhận dữ liệu: {e}", end='\n')
        return None, None, None

# Function to control the vehicle forward
def control_forward(new_sock, client_socket, traj, goal, ego, PI_acc, PI_yaw):
    # Control logic for forward movement
    while getDistance([ego.x, ego.y], goal) > 10:  # Loop until the vehicle is within 1 meter of the goal
        # Calculate target point for look ahead
        target_point = traj.getTargetPoint([ego.x, ego.y])

        # Receive feedback (position data) from another program (e.g., GPS, IMU)
        ego.x, ego.y, ego.yaw = receive_feedback(client_socket)

        if ego.x is None:  # If no valid feedback is received, continue with the loop
            print("Không nhận được dữ liệu feedback, tiếp tục. \n")
            continue

        print(f"[FEEDBACK FORWARD] X: {ego.x:.2f}, Y: {ego.y:.2f}, Yaw: {ego.yaw:.2f} \n")

        # Calculate error for velocity and yaw
        vel_err = ego.vel  # Constant velocity for forward
        acc = PI_acc.control(vel_err)

        # Calculate yaw error as the difference between target yaw and current yaw
        yaw_err = math.atan2(target_point[1] - ego.y, target_point[0] - ego.x) - ego.yaw

        # Normalize yaw error to be within [-pi, pi]
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi  # Normalize between -pi and pi

        # If yaw error is very small (meaning we are going straight), set delta to 0
        if abs(yaw_err) < 0.05:  # Tolerance threshold for yaw error
            delta = 0
        else:
            delta = PI_yaw.control(yaw_err)

        delta = max(-math.radians(30), min(math.radians(30), delta))

        # Check if the vehicle has reached the goal (within a 1-meter threshold)
        if getDistance([ego.x, ego.y], goal) <= 10:
            print("Xe đã đến điểm cuối. Dừng lại. \n")
            ego.vel = 0  # Set velocity to 0 to stop the vehicle
            delta = 0  # Ensure the steering angle is also set to 0

        # Send control command (velocity and steering angle) over the new TCP connection
        send_control_command(new_sock, ego.vel, delta)

        # Print current target point, velocity and steering angle
        print(f"Target Point: {target_point} \n")
        print(f"Current Velocity: {ego.vel} cm/s \n")
        print(f"Steering Angle: {math.degrees(delta):.2f} degrees \n")


# Function to control the vehicle reverse (lùi)
def control_reverse(new_sock, client_socket, traj, goal, ego, PI_acc, PI_yaw):
    # Control logic for reverse movement
    while getDistance([ego.x, ego.y], goal) > 10:  # Loop until the vehicle is within 1 meter of the goal
        # Calculate target point for look ahead
        target_point = traj.getTargetPoint([ego.x, ego.y])

        # Receive feedback (position data) from another program (e.g., GPS, IMU)
        ego.x, ego.y, ego.yaw = receive_feedback(client_socket)

        if ego.x is None:  # If no valid feedback is received, continue with the loop
            print("Không nhận được dữ liệu feedback, tiếp tục. \n")
            continue
            
        print(f"[FEEDBACK REVERSE] X: {ego.x:.2f}, Y: {ego.y:.2f}, Yaw: {ego.yaw:.2f} \n")

        # Calculate error for velocity and yaw
        vel_err = ego.vel  # Constant negative velocity for reverse
        acc = PI_acc.control(vel_err)

        # Calculate yaw error as the difference between target yaw and current yaw
        yaw_err = math.atan2(target_point[1] - ego.y, target_point[0] - ego.x) - ego.yaw

        # Flip for reverse (reverse vehicle has the opposite yaw control)
        yaw_err = (yaw_err + math.pi) % (2 * math.pi)

        # Normalize yaw error to be within [-pi, pi]
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi  # Normalize between -pi and pi

        # If yaw error is very small (meaning we are going straight), set delta to 0
        if abs(yaw_err) < 0.05:  # Tolerance threshold for yaw error
            delta = 0
        else:
            delta = PI_yaw.control(yaw_err) * -1

        # Ensure delta is within the allowed range [-30, 30] degrees
        delta = max(-math.radians(30), min(math.radians(30), delta))

        # Check if the vehicle has reached the goal (within a 1-meter threshold)
        if getDistance([ego.x, ego.y], goal) <= 10:
            print("Xe đã đến điểm cuối. Dừng lại.", end='\n')
            ego.vel = 0  # Set velocity to 0 to stop the vehicle
            delta = 0  # Ensure the steering angle is also set to 0

        # Send control command (velocity and steering angle) over the new TCP connection
        send_control_command(new_sock, ego.vel, delta)

        # Print current target point, velocity and steering angle
        print(f"Target Point: {target_point} \n")
        print(f"Current Velocity: {ego.vel} cm/s \n")
        print(f"Steering Angle: {math.degrees(delta):.2f} degrees \n")

def split_trajectory(traj_data):
    segment = []
    current_trajectory = []

    x1, y1, yaw1 = traj_data[0]

    for i in range(1, len(traj_data)):
        x2, y2, yaw2 = traj_data[i]
        dx = x2 - x1
        dy = y2 - y1
        dx_yaw = math.cos(yaw2)
        dy_yaw = math.sin(yaw2)
        dot_product = dx * dx_yaw + dy * dy_yaw
        magnitude_move = math.sqrt(dx ** 2 + dy ** 2)
        magnitude_yaw = math.sqrt(dx_yaw ** 2 + dy_yaw ** 2)
        cos_theta = dot_product / (magnitude_move * magnitude_yaw + 1e-6)

        is_forward = cos_theta > 0

        if not current_trajectory:
            current_trajectory.append((x1, y1, yaw1))
            current_direction = is_forward
        elif current_direction != is_forward:
            current_trajectory.append((x1, y1, yaw1))
            segment.append(current_trajectory)
            current_trajectory = [(x1, y1, yaw1)]
            current_direction = is_forward

        current_trajectory.append((x2, y2, yaw2))
        x1, y1, yaw1 = x2, y2, yaw2

    if current_trajectory:
        segment.append(current_trajectory)

    return segment

def detect_segment_direction(segment):
    for j in range(1, len(segment)):
        x1, y1, yaw1 = segment[j - 1]
        x2, y2, yaw2 = segment[j]
        dx = x2 - x1
        dy = y2 - y1
        dx_yaw = math.cos(yaw2)
        dy_yaw = math.sin(yaw2)
        dot_product = dx * dx_yaw + dy * dy_yaw
        magnitude_move = math.sqrt(dx ** 2 + dy ** 2)
        magnitude_yaw = math.sqrt(dx_yaw ** 2 + dy_yaw ** 2)
        cos_theta = dot_product / (magnitude_move * magnitude_yaw + 1e-6)
        return 'forward' if cos_theta > 0 else 'reverse'
    return 'undefined'

def check_and_process_segments(segments):
    for i, seg in enumerate(segments):
        print(f"Segment {i + 1}:")
        for point in seg:
            if not isinstance(point, tuple) or len(point) != 3:
                print(f"  ⚠️ Invalid point: {point}", end='\n')
        move_type = detect_segment_direction(seg)
        print(f"  → Direction: {move_type}", end='\n')
    print()

def vehicle_move(traj_data, client_socket, new_sock, forward=True):
    # Initialize the vehicle, trajectory, and controllers
    traj_x = [point[0] for point in traj_data]
    traj_y = [point[1] for point in traj_data]
    traj_yaw = [point[2] for point in traj_data]

    start_x, start_y, start_yaw = traj_x[0], traj_y[0], traj_yaw[0]
    target_vel = 10 if forward else -10  # Positive for forward, negative for reverse
    traj = Trajectory(traj_x, traj_y, traj_yaw)
    goal = traj.getPoint(len(traj_x) - 1)

    ego = Vehicle(start_x, start_y, start_yaw, vel=target_vel)
    PI_acc = PI()
    PI_yaw = PI()

    # Call the appropriate control function (forward or reverse)
    if forward:
        control_forward(new_sock, client_socket, traj, goal, ego, PI_acc, PI_yaw)
    else:
        control_reverse(new_sock, client_socket, traj, goal, ego, PI_acc, PI_yaw)

def control():
    # Load trajectory from file
    json_file = "path_data.json"
    traj_data = load_trajectory_from_file(json_file)

    if not traj_data:
        print("Quỹ đạo không được tải về, dừng chương trình.", end='\n')
        return

    # Create a new TCP connection for communication
    FB_IP = "127.0.0.1"
    FB_PORT = 5015
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((FB_IP, FB_PORT))
        sock.listen(1)
        print(f"Đang lắng nghe kết nối từ chương trình khác tại {FB_IP}:{FB_PORT}...", end='\n')
        client_socket, client_address = sock.accept()
        print(f"Đã kết nối với {client_address}", end='\n')
        
        # Initialize ego vehicle with initial position
        data = client_socket.recv(1024).decode("utf-8").strip()
        if data:
            x, y, yaw = map(float, data.split(","))
            ego = Vehicle(x, y, yaw)
            print(f"[FEEDBACK INITIAL] X: {ego.x:.2f}, Y: {ego.y:.2f}, Yaw: {ego.yaw:.2f}", end='\n')
        
    except Exception as e:
        print(f"Lỗi kết nối TCP: {e}", end='\n')
        return
    
    # Create a new socket to send control data to another TCP server
    Ctrl_IP = "192.168.1.137"
    Ctrl_PORT = 5001
    # Ctrl_IP = "127.0.0.1"
    # Ctrl_PORT = 5002
    try:
        new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_sock.connect((Ctrl_IP, Ctrl_PORT))
        print(f"Đã kết nối với {Ctrl_IP}:{Ctrl_PORT}")
    except Exception as e:
        print(f"Lỗi kết nối đến server mới: {e}")
        return

    # Split the trajectory into segments
    sub_traj = split_trajectory(traj_data)

    # Process each segment
    for i, seg in enumerate(sub_traj):
        direction = detect_segment_direction(seg)
        if direction == 'forward':
            print(f"Segment {i+1}: Di chuyển tới")
            vehicle_move(seg, client_socket, new_sock, forward=True)
        else:
            print(f"Segment {i+1}: Di chuyển lùi")
            vehicle_move(seg, client_socket, new_sock, forward=False)

    # Close connections after all segments are processed
    new_sock.close()
    client_socket.close()
    sock.close()
    print("Đã đóng kết nối TCP.")

if __name__ == "__main__":
    control()