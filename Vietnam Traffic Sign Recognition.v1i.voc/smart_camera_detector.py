import cv2
import pickle
import numpy as np
import os
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

# 1. Nạp bộ tri thức
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_hog.pkl')

with open(model_path, 'rb') as f:
    data = pickle.load(f)
    X, y = data['features'], data['labels']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Bước 1: Tiền xử lý để tìm vật thể (Dùng lọc màu đỏ - đặc trưng biển báo cấm)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Lọc màu đỏ (biển báo cấm thường có viền đỏ)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.addWeighted(cv2.inRange(hsv, lower_red1, upper_red1), 1.0,
                           cv2.inRange(hsv, lower_red2, upper_red2), 1.0, 0)

    # Bước 2: Tìm các đường bao (Contours) của vật thể màu đỏ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Chỉ xử lý các vật thể đủ lớn
            x, y, w, h = cv2.boundingRect(cnt)

            # Cắt vùng nghi vấn (ROI)
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0: continue

            # Bước 3: Dùng HOG để kiểm tra vùng này có phải biển báo không
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized_roi = cv2.resize(gray_roi, (64, 64))

            fd = hog(resized_roi, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), feature_vector=True)

            # Dự đoán
            prediction = model.predict([fd])[0]

            # Bước 4: Vẽ khung và hiện tên nhãn ngay tại vị trí tìm thấy
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(prediction), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Smart Detection (Anywhere)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()