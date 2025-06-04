import numpy as np
import matplotlib.pyplot as plt
import socket
from hybridAStar_v4 import run
from hybridAStar_v4 import calculateMapParameters
from hybridAStar_v4 import create_map
from combine_PurePursuit import control
import time

TCP_IP = "127.0.0.1"
TCP_PORT = 5013
BUFFER_SIZE = 1024

# Initialize start position variables
x_start = None
y_start = None
yaw_start = None

def receive_start_position():
    global x_start, y_start, yaw_start 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"[START RECEIVER] Đang lắng nghe tại {TCP_IP}:{TCP_PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"[START RECEIVER] Đã kết nối từ: {addr}")
            data = conn.recv(BUFFER_SIZE).decode("utf-8")
            if data:
                parts = data.strip().split(",")
                if len(parts) >= 4 and parts[0] == "START":  # Check for START prefix
                    x_start, y_start, yaw_start = map(float, parts[1:4])  # Skip START and take next 3 values
                    print(f"[START RECEIVER] Vị trí bắt đầu nhận được: X={x_start:.2f}, Y={y_start:.2f}, Yaw={yaw_start:.2f}")
                else:
                    print(f"[START RECEIVER] Dữ liệu không hợp lệ: {data}")

def main():
    # First receive the start position
    receive_start_position()
    
    # Wait until we have valid start position
    while x_start is None or y_start is None or yaw_start is None:
        print("Waiting for start position...")
        time.sleep(1)
    
    # Init chướng ngại vật và map
    obstacleX, obstacleY = create_map()
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))

    # Set điểm xuất phát
    s = [x_start, y_start, np.deg2rad(yaw_start)]
    
    # Set goal position
    g = [220, 130, np.deg2rad(-89)]
    
    # Run the path planning and control
    run(s, g, mapParameters, plt)
    control()

if __name__ == "__main__":
    main()


