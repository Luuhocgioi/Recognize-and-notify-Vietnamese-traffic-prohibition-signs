import cv2
import numpy as np
from skimage.feature import hog
import joblib

# 1. Đánh thức "bộ não" AI
print("Đang tải mô hình SVM...")
svm_model = joblib.load('svm_traffic_model.pkl')
print("Tải thành công! Bật camera...")

# 2. Mở Camera (Số 0 là camera mặc định của laptop)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- BƯỚC A: TÌM BIỂN BÁO BẰNG LỌC MÀU ---
    # Đổi sang hệ màu HSV để lọc màu chuẩn hơn (không bị ảnh hưởng bởi ánh sáng)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa dải màu Đỏ
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Định nghĩa dải màu Xanh dương
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Gộp 2 mặt nạ màu lại
    mask_combined = cv2.bitwise_or(mask_red, mask_blue)

    # Tìm đường viền (contours) của các mảng màu vừa lọc
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- BƯỚC B: CẮT BIỂN BÁO VÀ ĐƯA CHO AI DỰ ĐOÁN ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # Lọc bỏ các đốm màu nhỏ xíu (bị nhiễu)
            x, y, w, h = cv2.boundingRect(cnt)

            # Biển báo thường là hình vuông/tròn/tam giác nên chiều rộng và cao xấp xỉ nhau
            ratio = w / float(h)
            if 0.7 <= ratio <= 1.3:

                # Cắt cái biển báo ra
                roi = frame[max(0, y):y + h, max(0, x):x + w]

                try:
                    # Đưa về ảnh xám -> Ép size 64x64 -> Trích xuất HOG
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (64, 64))
                    features = hog(roi_resized, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

                    # Gọi SVM dự đoán
                    prediction = svm_model.predict([features])[0]

                    # Vẽ khung xanh lá cây và in tên biển báo lên màn hình
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # prediction chính là tên thư mục mà bạn đã dùng để train
                    cv2.putText(frame, str(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    pass

    cv2.imshow("Nhan dien Bien Bao - HOG + SVM", frame)

    # Nhấn phím 'q' trên bàn phím để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()