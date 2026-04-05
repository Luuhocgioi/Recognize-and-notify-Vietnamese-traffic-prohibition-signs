import os
import cv2
import xml.etree.ElementTree as ET

INPUT_FOLDER = 'train'
OUTPUT_FOLDER = 'Vietnam Traffic Sign Recognition.v1i.voc/cropped_signs'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.xml'):
        xml_path = os.path.join(INPUT_FOLDER, filename)

        # Thử tìm file ảnh cùng tên (ví dụ: abc.xml -> abc.jpg)
        image_path = xml_path.replace('.xml', '.jpg')

        if not os.path.exists(image_path):
            # Nếu không thấy .jpg, thử tìm .png
            image_path = xml_path.replace('.xml', '.png')

        img = cv2.imread(image_path)
        if img is None:
            print(f"Không tìm thấy ảnh cho file: {filename}")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for i, obj in enumerate(root.findall('object')):
            label = obj.find('name').text
            xmlbox = obj.find('bndbox')

            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)

            # Cắt và Resize về 64x64 để chuẩn hóa cho HOG
            crop_img = img[ymin:ymax, xmin:xmax]
            if crop_img.size == 0: continue
            resized_img = cv2.resize(crop_img, (64, 64))

            label_dir = os.path.join(OUTPUT_FOLDER, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            save_path = os.path.join(label_dir, f"{i}_{filename.replace('.xml', '.jpg')}")
            cv2.imwrite(save_path, resized_img)

print("Xong! Bạn kiểm tra thư mục 'cropped_signs' nhé.")