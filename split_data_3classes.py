import os
import random
import shutil

# ==========================================
# 📁 1. โฟลเดอร์ปลายทาง (Dataset หลักของโปรเจกต์)
# ==========================================
TARGET_BASE_DIR = r"C:\Users\Admin\Desktop\Project\pillwebcam\bbox_split"

# ==========================================
# 📁 2. รายชื่อโฟลเดอร์ยาทั้ง 3 ชนิดของคุณ
# ⚠️ เปลี่ยน path ให้ตรงกับโฟลเดอร์ยาแต่ละชนิดของคุณนะครับ
# ==========================================
SOURCE_FOLDERS = [
    r"C:\Users\Admin\Desktop\Project\pillwebcam\capsule",  # ยาชนิดที่ 1 (เช่น วงกลม/แบบที่ 1)
    r"C:\Users\Admin\Desktop\Project\pillwebcam\oval",  # ยาชนิดที่ 2 (เช่น วงรี/แบบที่ 2)
    r"C:\Users\Admin\Desktop\Project\pillwebcam\square"   # ยาชนิดที่ 3 (เช่น สี่เหลี่ยม/แบบที่ 3)
]

TRAIN_RATIO = 0.8  # สัดส่วน Train 80%

# ---------------------------------------------------------
# สร้างโฟลเดอร์ปลายทางรอไว้ก่อน
train_img_dir = os.path.join(TARGET_BASE_DIR, "images", "train")
test_img_dir = os.path.join(TARGET_BASE_DIR, "images", "test")
train_lbl_dir = os.path.join(TARGET_BASE_DIR, "labels", "train")
test_lbl_dir = os.path.join(TARGET_BASE_DIR, "labels", "test")

for d in [train_img_dir, test_img_dir, train_lbl_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

valid_extensions = ('.jpg', '.jpeg', '.png')

# ฟังก์ชันสำหรับก๊อปปี้ไฟล์
def copy_files(file_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    for img_filename in file_list:
        base_name = os.path.splitext(img_filename)[0]
        txt_filename = f"{base_name}.txt"
        
        src_img = os.path.join(src_img_dir, img_filename)
        src_lbl = os.path.join(src_lbl_dir, txt_filename)
        
        dst_img = os.path.join(dest_img_dir, img_filename)
        dst_lbl = os.path.join(dest_lbl_dir, txt_filename)
        
        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# ==========================================
# 🚀 เริ่มการสุ่มทีละโฟลเดอร์
# ==========================================
print("กำลังเริ่มสุ่มแบ่งข้อมูลทีละชนิด (Train 80% / Test 20%)...")
print("=" * 60)

total_train = 0
total_test = 0

for folder_path in SOURCE_FOLDERS:
    folder_name = os.path.basename(folder_path)
    src_img_dir = os.path.join(folder_path, "images")
    src_lbl_dir = os.path.join(folder_path, "labels")
    
    # เช็กว่ามีโฟลเดอร์นี้อยู่จริงไหม
    if not os.path.exists(src_img_dir):
        print(f"⚠️ ข้ามโฟลเดอร์ '{folder_name}' (หาโฟลเดอร์ images ไม่เจอ)")
        continue

    # ดึงรายชื่อรูปภาพ
    images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(valid_extensions)]
    
    if len(images) == 0:
        continue
        
    # 🎲 สุ่มสลับไฟล์ของยาชนิดนี้
    random.shuffle(images)
    
    # คำนวณจุดตัด 80/20
    split_idx = int(len(images) * TRAIN_RATIO)
    train_files = images[:split_idx]
    test_files = images[split_idx:]
    
    # สั่งก๊อปปี้
    copy_files(train_files, src_img_dir, src_lbl_dir, train_img_dir, train_lbl_dir)
    copy_files(test_files, src_img_dir, src_lbl_dir, test_img_dir, test_lbl_dir)
    
    # เก็บสถิติ
    total_train += len(train_files)
    total_test += len(test_files)
    
    print(f"📦 ชนิดยา: {folder_name} (ทั้งหมด {len(images)} รูป)")
    print(f"   -> โควต้า Train: {len(train_files)} รูป")
    print(f"   -> โควต้า Test : {len(test_files)} รูป")
    print("-" * 60)

print("🎉 เสร็จสมบูรณ์! ข้อมูลถูกเทรวมกันใน bbox_split เรียบร้อยแล้ว")
print(f"📊 สรุปรวม Dataset ทั้งหมด: Train = {total_train} รูป | Test = {total_test} รูป")