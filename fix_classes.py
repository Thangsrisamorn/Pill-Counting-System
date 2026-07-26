import os

LABEL_DIR = "labels"

for filename in os.listdir(LABEL_DIR):

    if not filename.endswith(".txt"):
        continue

    image_number = int(filename.split("_")[1].replace(".txt", ""))

    if image_number <= 410:
        new_class = 0      
    else:
        new_class = 1      

    path = os.path.join(LABEL_DIR, filename)

    new_lines = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            parts[0] = str(new_class)

            new_lines.append(" ".join(parts))

    with open(path, "w") as f:
        f.write("\n".join(new_lines))

print("แก้ไขคลาสทั้งหมดเรียบร้อย")
