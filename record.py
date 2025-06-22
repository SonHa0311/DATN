import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở stream!")
    exit()

# Đọc frame đầu tiên để xác định kích thước crop
ret, frame = cap.read()
height, width = frame.shape[:2]

# Crop từ x=320 đến x=1880
x_start, x_end = 320, 1920
cropped_width = x_end - x_start

# Khởi tạo VideoWriter cho birdseye_cropped
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('birdseye_view_only.mp4', fourcc, 20.0, (cropped_width, height))

# Vòng lặp xử lý
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được frame từ stream!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Chuyển đổi sang Bird's Eye View
    pts_src = np.float32([[404, 130], [1816, 104], [1852, 1018], [386, 966]])
    pts_dst = np.float32([[404, 104], [1816, 104], [1816, 1018], [404, 1018]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    birdseye_frame = cv2.warpPerspective(frame, matrix, (width, height))

    # Cắt theo trục x
    birdseye_cropped = birdseye_frame[:, x_start:x_end]

    # Hiển thị
    cv2.imshow("Bird's Eye View", birdseye_cropped)

    # Ghi video Bird's Eye View đã cắt
    out.write(birdseye_cropped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng
cap.release()
out.release()
cv2.destroyAllWindows()
