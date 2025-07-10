import socket
import json
import serial

# Cấu hình server TCP
IP_CONTROL = "192.168.1.137"
PORT_CONTROL = 5004

# Cấu hình UART với STM32
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)

def map_velocity(v):
    # v = max(-0.1, min(0.1, v))
    con = 4400
    if v == 0.1:
        mapped = con
    if v == -0.1:
        mapped = -con
    if v == 0:
        mapped = 0
    # mapped = int( v * delta )
    return f"{mapped:+05d}"  # +0000 hoặc -0000

def map_steering(delta):
    delta = -delta + 80
    return int(delta)

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((IP_CONTROL, PORT_CONTROL))
        server.listen(1)
        print(f"File 3 đang lắng nghe tại {IP_CONTROL}:{PORT_CONTROL}")

        conn, addr = server.accept()
        with conn:
            print(f"Kết nối từ File 2: {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break

                try:
                    command = data.decode('utf-8').strip()
                    print(f"Nhận lệnh: {command}")
    
                    v_str, delta_str = command.split(",")
                    v = float(v_str)
                    delta = float(delta_str)

                    motor_val = map_velocity(v)
                    steer_val = map_steering(delta)

                    motor_cmd = f"M{motor_val} "
                    steer_cmd = f"S{steer_val} "

                    ser.write(motor_cmd.encode())
                    ser.write(steer_cmd.encode())

                    print(motor_cmd.strip())
                    print(steer_cmd.strip())

                except Exception as e:
                    print(f"Lỗi xử lý lệnh: {e}")


def main():
    try:
        start_server()
    except KeyboardInterrupt:
        print("Dừng chương trình")
    finally:
        ser.close()

if __name__ == '__main__':
    main()
