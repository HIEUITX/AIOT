from ultralytics import YOLO
import cv2
import math

# 1. Tải mô hình Pose Estimation chuyên dụng của YOLOv8
model = YOLO("yolov8n-pose.pt")

# Khởi chạy camera laptop
cap = cv2.VideoCapture(0)

print("Đang khởi chạy hệ thống nhận diện khung xương... Nhấn ESC trên cửa sổ camera để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Dự đoán và trích xuất điểm mốc cơ thể (Khung xương)
    results = model(frame, conf=0.5, verbose=False)
    
    # Lấy ảnh đã vẽ sẵn khung xương mặc định từ YOLOv8
    annotated = results[0].plot()

    # Mặc định ban đầu trạng thái là bình thường
    status_text = "Trang thai: Binh thuong"
    status_color = (0, 255, 0) # Màu xanh lá
    alert_triggered = False

    # 3. Thuật toán phân tích logic dựa trên Tọa độ Keypoints (Nếu phát hiện thấy người)
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        # Lấy danh sách điểm nút của người đầu tiên phát hiện được
        # Cấu trúc index keypoints của YOLOv8-Pose: 
        # 0: Mũi, 5: Vai trái, 6: Vai phải, 9: Cổ tay trái, 10: Cổ tay phải, 11: Hông trái, 12: Hông phải
        kp = results[0].keypoints.xy[0].cpu().numpy()

        if len(kp) > 12:
            # --- LOGIC 1: PHÁT HIỆN GIƠ TAY ---
            # Nếu cổ tay trái (9) hoặc cổ tay phải (10) cao hơn vai tương ứng (5, 6)
            # Lưu ý: Trong OpenCV, trục Y hướng xuống dưới, nên Y càng nhỏ thì vị trí càng cao
            left_wrist_y = kp[9][1]
            left_shoulder_y = kp[5][1]
            right_wrist_y = kp[10][1]
            right_shoulder_y = kp[6][1]

            # Kiểm tra xem các điểm có hợp lệ không (khác 0)
            if (left_wrist_y > 0 and left_shoulder_y > 0 and left_wrist_y < left_shoulder_y) or \
               (right_wrist_y > 0 and right_shoulder_y > 0 and right_wrist_y < right_shoulder_y):
                status_text = "CANH BAO: CO NGUOI GIO TAY!"
                status_color = (0, 165, 255) # Màu Cam
                alert_triggered = True

            # --- LOGIC 2: PHÁT HIỆN NGHIÊNG NGƯỜI ---
            # Tính góc nghiêng dựa vào điểm Mũi (0) và điểm giữa hai Hông (11, 12) để tạo thành trục sống lưng
            nose_x, nose_y = kp[0][0], kp[0][1]
            hip_left_x, hip_left_y = kp[11][0], kp[11][1]
            hip_right_x, hip_right_y = kp[12][0], kp[12][1]

            if nose_x > 0 and hip_left_x > 0 and hip_right_x > 0:
                # Tâm hông
                hip_center_x = (hip_left_x + hip_right_x) / 2
                hip_center_y = (hip_left_y + hip_right_y) / 2

                # Tính góc lệch so với phương thẳng đứng
                dx = nose_x - hip_center_x
                dy = hip_center_y - nose_y # đảo chiều y vì y màn hình quay xuống
                
                if dy != 0:
                    angle = math.degrees(math.atan(abs(dx) / dy))
                    # Nếu góc nghiêng cơ thể lệch quá 20 độ
                    if angle > 20:
                        status_text = f"CANH BAO: NGUOI NGHIENG ({int(angle)} deg)!"
                        status_color = (0, 0, 255) # Màu Đỏ nguy hiểm
                        alert_triggered = True

    # 4. GIAO DIỆN ĐỒ HỌA THÔNG BÁO TRÊN MÀN HÌNH (UI OVERLAY)
    h, w, _ = annotated.shape

    # Nếu có cảnh báo (Giơ tay / Nghiêng người), vẽ thêm một khung viền Flash nhấp nháy màu đỏ/cam quanh camera
    if alert_triggered:
        cv2.rectangle(annotated, (0, 0), (w, h), status_color, 8)
    
    # Vẽ thanh Banner nền đen phía trên để hiển thị chữ rõ ràng
    cv2.rectangle(annotated, (10, 10), (460, 60), (20, 20, 20), -1)
    
    # Viết chữ thông báo trạng thái động lên màn hình
    cv2.putText(
        annotated, 
        status_text, 
        (20, 42), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.65, 
        status_color, 
        2, 
        cv2.LINE_AA
    )

    # Hiển thị cửa sổ giao diện đã được bổ sung thông báo
    cv2.imshow("Pose Estimation", annotated)

    key = cv2.waitKey(1)
    if key == 27: # Nhấn phím ESC để tắt
        break

cap.release()
cv2.destroyAllWindows()