import os
import cv2

# ==========================
# ตั้งค่าโฟลเดอร์
# ==========================

IMAGE_DIR = "images"
LABEL_DIR = "labels"
OUTPUT_DIR = "bounding_preview"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# สี Bounding Box
# ==========================

COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0)
}


# ==========================
# หา Label ที่ตรงกับรูป
# ==========================

def find_label(image_name):

    name_without_ext = os.path.splitext(image_name)[0]


    # กรณีชื่อเหมือนกัน
    label_path = os.path.join(
        LABEL_DIR,
        name_without_ext + ".txt"
    )

    if os.path.exists(label_path):
        return label_path


    # กรณี Roboflow
    prefix = name_without_ext + "_jpg.rf"

    for file in os.listdir(LABEL_DIR):

        if file.startswith(prefix) and file.endswith(".txt"):
            return os.path.join(
                LABEL_DIR,
                file
            )

    return None



# ==========================
# วนอ่านรูปทั้งหมด
# ==========================

for image_file in os.listdir(IMAGE_DIR):

    if not image_file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue


    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )


    label_path = find_label(image_file)


    if label_path is None:
        print("ไม่พบ Label :", image_file)
        continue



    img = cv2.imread(image_path)


    if img is None:
        print("อ่านรูปไม่ได้ :", image_file)
        continue


    h, w, _ = img.shape


    count = 0   # ตัวนับ Bounding Box


    with open(label_path, "r") as f:

        labels = f.readlines()



    for label in labels:

        data = label.strip().split()

        if len(data) != 5:
            continue


        cls = int(data[0])

        x_center = float(data[1])
        y_center = float(data[2])
        box_w = float(data[3])
        box_h = float(data[4])


        # ==========================
        # YOLO -> Pixel
        # ==========================

        x1 = int(
            (x_center - box_w/2) * w
        )

        y1 = int(
            (y_center - box_h/2) * h
        )

        x2 = int(
            (x_center + box_w/2) * w
        )

        y2 = int(
            (y_center + box_h/2) * h
        )


        color = COLORS.get(
            cls,
            (0,255,0)
        )


        # วาด Bounding Box อย่างเดียว

        cv2.rectangle(
            img,
            (x1,y1),
            (x2,y2),
            color,
            2
        )


        count += 1



    # ==========================
    # แสดงจำนวนเม็ดทั้งหมด
    # ==========================

    cv2.putText(
        img,
        f"Count: {count}",
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0,0,255),
        3
    )


    # Save

    output_path = os.path.join(
        OUTPUT_DIR,
        image_file
    )


    cv2.imwrite(
        output_path,
        img
    )


    print(
        f"สร้าง Preview: {image_file} | Count = {count}"
    )



print("\nเสร็จแล้ว ดูรูปได้ที่ bounding_preview")