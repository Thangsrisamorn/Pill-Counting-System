import os

LABEL_DIR = "labels"

for file in os.listdir(LABEL_DIR):

    if file.endswith(".txt"):

        # ตัดส่วน _jpg.rf.xxx.txt
        new_name = file.split("_jpg.rf")[0] + ".txt"

        old_path = os.path.join(LABEL_DIR, file)
        new_path = os.path.join(LABEL_DIR, new_name)

        os.rename(old_path, new_path)

        print(f"{file} -> {new_name}")

print("Rename เสร็จแล้ว")