import os
import cv2

# --- BẠN CẦN CHỈNH SỬA 2 DÒNG NÀY ---
# Đảm bảo đường dẫn này trỏ đúng vào thư mục images và labels bạn vừa giải nén
images_dir = "roboflow_dataset/train/images"
labels_dir = "roboflow_dataset/train/labels"
# -----------------------------------

# Thư mục đích để chứa ảnh sau khi cắt
output_dir = "dataset_classification/train"
os.makedirs(output_dir, exist_ok=True)

print("Đang tiến hành đọc tọa độ và cắt ảnh... Vui lòng đợi!")
count = 0

for img_name in os.listdir(images_dir):
    if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue

    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    # Đọc ảnh gốc bằng OpenCV
    img = cv2.imread(img_path)
    if img is None or not os.path.exists(label_path): continue
    h_img, w_img, _ = img.shape

    # Đọc tọa độ từ file txt
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        data = line.strip().split()
        class_id = data[0]

        # Chuyển đổi tọa độ tỷ lệ (%) của YOLO sang tọa độ Pixel thực tế của ảnh
        x_center, y_center, w, h = map(float, data[1:5])
        x1 = int((x_center - w / 2) * w_img)
        y1 = int((y_center - h / 2) * h_img)
        x2 = int((x_center + w / 2) * w_img)
        y2 = int((y_center + h / 2) * h_img)

        # Đảm bảo tọa độ không bị tràn ra ngoài viền ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Bỏ qua nếu khung cắt quá nhỏ (nhiễu)
        if x2 - x1 < 10 or y2 - y1 < 10: continue

        # Cắt riêng biển báo ra
        cropped_img = img[y1:y2, x1:x2]

        # Lưu vào đúng thư mục tương ứng với mã biển báo
        class_dir = os.path.join(output_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{img_name.split('.')[0]}_crop_{idx}.jpg")
        cv2.imwrite(save_path, cropped_img)
        count += 1

print(f"XONG! Đã cắt thành công {count} biển báo.")
print(f"Hãy kiểm tra thư mục: {output_dir}")