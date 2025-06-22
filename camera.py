import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở stream!")


while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame từ stream!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        # Lấy kích thước khung hình
        height, width = frame.shape[:2]

        # Chuyển đổi ảnh sang Bird's Eye View
        pts_src = np.float32([[404, 130], [1816, 104], [1852,1018], [386, 966]])
        pts_dst = np.float32([[404, 104], [1816, 104], [1816, 1018], [404, 1018]])
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        birdseye_frame = cv2.warpPerspective(frame, matrix, (width, height))
        birdseye_gray = cv2.warpPerspective(gray, matrix, (width, height))
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        birdseye_frame = cv2.warpPerspective(frame, matrix, (width, height))

        # Cắt theo trục x từ 350 đến 1920
        frame_cropped = frame[:, 320:1880]
        birdseye_cropped = birdseye_frame[:, 320:1880]

        # Hiển thị khung hình
        cv2.imshow("Original Stream", frame_cropped)
        cv2.imshow("Bird's Eye View", birdseye_cropped)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


