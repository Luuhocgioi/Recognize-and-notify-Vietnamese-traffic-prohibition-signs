import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
import joblib

# Đường dẫn tới thư mục ảnh bạn vừa cắt thành công
dataset_path = 'dataset_classification/train'

X = []  # Chứa đặc trưng HOG (Câu hỏi)
y = []  # Chứa tên biển báo (Đáp án)

print("Đang đọc ảnh và trích xuất đặc trưng HOG. Vui lòng đợi...")

# Tự động lấy danh sách các thư mục con (0, 1, 2...) làm nhãn (label)
classes = os.listdir(dataset_path)

for label in classes:
    folder_path = os.path.join(dataset_path, label)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh đen trắng để HOG xử lý dễ hơn

        if img is not None:
            # 1. Ép tất cả ảnh về cùng 1 size 64x64
            img_resized = cv2.resize(img, (64, 64))

            # 2. Trích xuất đặc trưng HOG
            features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

            X.append(features)
            y.append(label)

# Chuyển list thành mảng numpy để SVM hiểu được
X = np.array(X)
y = np.array(y)

print(f"Đã trích xuất xong! Tổng số ảnh AI sẽ học: {len(X)}")
print("--------------------------------------------------")
print("Bắt đầu huấn luyện mô hình SVM (Quá trình này có thể mất 1-3 phút)...")

# 3. Tạo và Huấn luyện mô hình SVM
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X, y)

# 4. Lưu lại "bộ não" sau khi học xong
joblib.dump(svm_model, 'svm_traffic_model.pkl')

print("Đã huấn luyện xong và lưu bộ não thành file 'svm_traffic_model.pkl'!")