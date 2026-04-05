import cv2
import pickle
import numpy as np
import os
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

# 1. Nạp bộ tri thức (model_hog.pkl)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_hog.pkl')

with open(model_path, 'rb') as f:
    data = pickle.load(f)
    X, y = data['features'], data['labels']

# Huấn luyện bộ phân loại nhanh
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# 2. Mở Camera (0 thường là webcam mặc định)
cap = cv2.VideoCapture(0)

print("Đang mở Camera... Đưa biển báo trước ống kính. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Tạo vùng quét ở giữa màn hình (Bounding Box giả lập)
    # Vì HOG cần ảnh gọn, bạn nên đưa biển báo vào ô vuông này
    h, w, _ = frame.shape
    start_point = (w // 2 - 100, h // 2 - 100)
    end_point = (w // 2 + 100, h // 2 + 100)
    cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 2)

    # Cắt vùng ảnh trong ô vuông để nhận diện
    roi = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    # Tiền xử lý vùng cắt (ROI)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))

    # Trích chọn đặc trưng HOG
    fd = hog(resized, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), feature_vector=True)

    # Dự đoán
    prediction = model.predict([fd])[0]

    # Hiển thị kết quả lên màn hình Camera
    cv2.putText(frame, f"Bien bao: {prediction}", (w // 2 - 100, h // 2 - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Nhan dien bien bao qua Camera", frame)

    # Nhấn 'q' để tắt camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()