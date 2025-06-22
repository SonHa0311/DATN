import cv2
import numpy as np
from pyapriltags import Detector

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở stream!")

# Khởi tạo detector AprilTag
detector = Detector(families='tag25h9')

# Gán tọa độ thực của các tag (đơn vị: mm)
real_world_coords = {
    3: (0, 0),
    2: (440, 0),
    4: (0, 260),
    5: (440, 260)
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được frame từ stream!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]

    # ===== Nhận diện AprilTags =====
    tags = detector.detect(gray)

    image_pts = []
    world_pts = []
    tag13_img_pos = None

    for tag in tags:
        corners = tag.corners.astype(int)
        center = tuple(tag.center.astype(int))

        # Vẽ khung tag
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)

        # Hiển thị ID
        cv2.putText(frame, f'ID:{tag.tag_id}', (center[0]-20, center[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Ghi lại điểm ảnh và tọa độ thực nếu tag có trong danh sách
        if tag.tag_id in real_world_coords:
            image_pts.append(tag.center)
            world_pts.append(real_world_coords[tag.tag_id])

        # Ghi lại vị trí ảnh của ID13
        if tag.tag_id == 13:
            tag13_img_pos = tag.center

    # ===== Homography để tính tọa độ thực =====
    if len(image_pts) == 4:
        image_pts_np = np.array(image_pts, dtype=np.float32)
        world_pts_np = np.array(world_pts, dtype=np.float32)

        H, status = cv2.findHomography(image_pts_np, world_pts_np)

        for tag in tags:
            if tag.tag_id in [2, 3, 4, 5]:
                pt = np.array([[tag.center[0]], [tag.center[1]], [1]])
                real_pos = np.dot(H, pt)
                real_pos /= real_pos[2]
                x_real, y_real = real_pos[0][0], real_pos[1][0]
                print(f"Tag ID: {tag.tag_id} - Real: ({x_real:.2f}, {y_real:.2f}) mm")

                # Vẽ lên khung hình
                cv2.putText(frame, f"({int(x_real)}, {int(y_real)})", 
                            (int(tag.center[0])-20, int(tag.center[1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 2)

        # Tính riêng ID13
        if tag13_img_pos is not None:
            pt = np.array([[tag13_img_pos[0]], [tag13_img_pos[1]], [1]])
            real_pos_homogeneous = np.dot(H, pt)
            real_pos_homogeneous /= real_pos_homogeneous[2]

            x_real, y_real = real_pos_homogeneous[0][0], real_pos_homogeneous[1][0]
            print(f"ID13 tọa độ thực: ({x_real:.2f}, {y_real:.2f}) mm")

            cv2.putText(frame, f"({int(x_real)}, {int(y_real)})", 
                        (int(tag13_img_pos[0])-20, int(tag13_img_pos[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # ===== Bird’s Eye View transform =====
    pts_src = np.float32([[404, 130], [1816, 104], [1852,1018], [386, 966]])
    pts_dst = np.float32([[404, 104], [1816, 104], [1816, 1018], [404, 1018]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    birdseye_frame = cv2.warpPerspective(frame, matrix, (width, height))

    # ===== Crop khung hình (trục x từ 320 đến 1880) =====
    cropped_frame = frame[:, 320:1880]
    cropped_birdseye = birdseye_frame[:, 320:1880]

    # ===== Hiển thị các khung hình =====
    cv2.imshow("Original Frame", cropped_frame)
    cv2.imshow("Bird's Eye View", cropped_birdseye)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
