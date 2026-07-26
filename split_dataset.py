import os
import shutil
import random

IMAGE_DIR = "images"      
LABEL_DIR = "labels"       
OUTPUT_DIR = "bbox_split"
TRAIN_RATIO = 0.8

for split in ["train", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print("จำนวนรูปทั้งหมด :", len(images))

random.shuffle(images)
split_index = int(len(images) * TRAIN_RATIO)

train_images = images[:split_index]
test_images = images[split_index:]

print("----------------")
print("Train :", len(train_images))
print("Test  :", len(test_images))
print("----------------")

def copy_data(files, split_folder):
    for img_file in files:
       
        src_img = os.path.join(IMAGE_DIR, img_file)
        dst_img = os.path.join(OUTPUT_DIR, "images", split_folder, img_file)
        shutil.copy(src_img, dst_img)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(LABEL_DIR, label_file)
        dst_label = os.path.join(OUTPUT_DIR, "labels", split_folder, label_file)
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f" ไม่พบไฟล์ Label สำหรับ: {img_file}")

copy_data(train_images, "train")
copy_data(test_images, "test")

print("แบ่งรูปและ Label สำเร็จ")
