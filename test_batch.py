import cv2
import glob
import os
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Admin\Desktop\Project\pillwebcam\runs\detect\Pill_Project\yolov8s_four_classes\weights\best.pt"
model = YOLO(MODEL_PATH)

input_folder = "test_images"
output_folder = "test_results"  

os.makedirs(output_folder, exist_ok=True)

image_files = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

if len(image_files) == 0:
    print(f" ไม่พบรูปภาพในโฟลเดอร์ '{input_folder}' กรุณาเอารูปไปใส่ไว้ก่อน")
    exit()

print(f" พบรูปภาพทั้งหมด {len(image_files)} รูป! กำลังเริ่มสแกนและบันทึกผลลัพธ์...")
print("-" * 60)

for i, img_path in enumerate(image_files):
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f" อ่านไฟล์ {img_path} ไม่ได้ ข้ามไปรูปถัดไป...")
        continue

    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot(labels=False, conf=False)

    boxes = results[0].boxes
    circle_count = 0
    capsule_count = 0
    oval_count = 0     
    square_count = 0   
    
    for box in boxes:
        class_id = int(box.cls[0])
        if class_id == 0:
            circle_count += 1
        elif class_id == 1:
            capsule_count += 1
        elif class_id == 2:
            oval_count += 1
        elif class_id == 3:
            square_count += 1
            
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (20, 20), (220, 170), (20, 20, 20), -1)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    cv2.rectangle(annotated_frame, (20, 20), (220, 170), (255, 255, 255), 2)
    
    cv2.putText(annotated_frame, f"Circle : {circle_count}", (35, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 255, 100), 1)
    cv2.putText(annotated_frame, f"Capsule: {capsule_count}", (35, 85), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 255, 255), 1)
    cv2.putText(annotated_frame, f"Oval   : {oval_count}", (35, 120), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 150, 100), 1)
    cv2.putText(annotated_frame, f"Square : {square_count}", (35, 155), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 100, 255), 1)

    file_name = os.path.basename(img_path)
    output_path = os.path.join(output_folder, f"result_{file_name}")
    cv2.imwrite(output_path, annotated_frame) 

    print(f" รูปที่ {i+1}/{len(image_files)} [{file_name}]")
    print(f"   -> สรุป: วงกลม {circle_count} | แคปซูล {capsule_count} | วงรี {oval_count} | สี่เหลี่ยม {square_count}")
    print(f"    บันทึกสำเร็จที่: {output_path}")
    print("-" * 60)

    cv2.namedWindow("Batch Image Tester", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Batch Image Tester", 1280, 720) 
    cv2.imshow("Batch Image Tester", annotated_frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print(" ยกเลิกการรันโปรแกรม")
        break

cv2.destroyAllWindows()
print("บันทึกภาพผลลัพธ์ครบทุกรูปเรียบร้อย")
