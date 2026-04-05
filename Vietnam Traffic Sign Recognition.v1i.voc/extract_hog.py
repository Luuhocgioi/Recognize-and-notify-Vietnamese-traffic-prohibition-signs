import os
import cv2
import pickle
import numpy as np
from skimage.feature import hog

# Chỉnh lại đường dẫn này cho đúng với vị trí file .py của bạn
# Nếu file .py nằm cùng cấp với thư mục cropped_signs thì chỉ cần 'cropped_signs'
DATA_PATH = 'cropped_signs'
OUTPUT_MODEL = 'model_hog.pkl'

features = []
labels = []

if not os.path.exists(DATA_PATH):
    print(f"Lỗi: Không tìm thấy thư mục {DATA_PATH}. Hãy kiểm tra lại đường dẫn!")
    exit()

print("Đang bắt đầu trích chọn đặc trưng HOG...")

for label_name in os.listdir(DATA_PATH):
    label_dir = os.path.join(DATA_PATH, label_name)
    if not os.path.isdir(label_dir): continue

    print(f" Đang xử lý nhãn: {label_name}")

    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Tiền xử lý: Chuyển xám và đảm bảo kích thước 64x64
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        # Trích chọn đặc trưng HOG
        fd = hog(gray, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2),
                 visualize=False, # Tắt visualize để chạy nhanh hơn khi train số lượng lớn
                 feature_vector=True)

        features.append(fd)
        labels.append(label_name)

# Lưu thành file pkl
with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump({'features': np.array(features), 'labels': np.array(labels)}, f)

print(f"Thành công! Đã trích xuất {len(features)} mẫu đặc trưng.")
print(f"Dữ liệu đã lưu vào: {OUTPUT_MODEL}")