import cv2
import pickle
import numpy as np
import os
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

# --- Bước 0: Xử lý đường dẫn tự động ---
# Lấy đường dẫn của thư mục chứa chính file test_model.py này
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'model_hog.pkl')
# Lưu ý: Thay tên file ảnh dưới đây bằng một file CÓ THẬT trong thư mục valid của bạn
test_img_name = 'ch0_20250430101655_20250430101955_f240.jpg'
test_img_path = os.path.join(base_dir, 'valid', test_img_name)

# --- Bước 1: Nạp bộ đặc trưng đã trích xuất ---
if not os.path.exists(model_path):
    print(f"Lỗi: Không tìm thấy file {model_path}. Hãy chạy extract_hog.py trước!")
    exit()

with open(model_path, 'rb') as f:
    data = pickle.load(f)
    X = data['features']
    y = data['labels']

print(f"Đã nạp bộ tri thức với {len(X)} mẫu đặc trưng.")

# --- Bước 2: Huấn luyện bộ phân loại KNN ---
# n_neighbors=3 nghĩa là lấy 3 ông hàng xóm giống nhất để biểu quyết kết quả
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# --- Bước 3: Đọc và Tiền xử lý ảnh Test ---
img = cv2.imread(test_img_path)
if img is None:
    print(f"Lỗi: Không tìm thấy ảnh test tại {test_img_path}")
    # Gợi ý: Thử in ra danh sách file trong thư mục valid để kiểm tra tên
    if os.path.exists(os.path.join(base_dir, 'valid')):
        print("Các file có trong thư mục valid:", os.listdir(os.path.join(base_dir, 'valid'))[:5])
    exit()

# Giữ lại ảnh gốc để hiển thị, xử lý trên ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (64, 64))

# --- Bước 4: Trích chọn đặc trưng HOG cho ảnh Test ---
fd = hog(resized, orientations=9, pixels_per_cell=(8, 8),
         cells_per_block=(2, 2), feature_vector=True)

# --- Bước 5: Dự đoán và Hiển thị ---
prediction = model.predict([fd])[0] # Lấy nhãn đầu tiên dự đoán được

print(f"Dự đoán biển báo: {prediction}")

# Ghi kết quả lên ảnh
text = f"Nhan dien: {prediction}"
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Kiem tra bien bao - Nhom [Ten_Cua_Ban]", img)
cv2.waitKey(0)
cv2.destroyAllWindows()