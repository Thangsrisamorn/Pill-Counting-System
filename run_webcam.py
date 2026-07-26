import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"D:\Project\pillwebcam\runs\detect\Pill_Project\yolov8s_four_classes_final-2\weights\best.pt"
model = YOLO(MODEL_PATH)

print("กำลังเชื่อมต่อกล้อง...")
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

cv2.namedWindow("Real-time Pill Counter", cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(" ไม่สามารถดึงภาพจากกล้องได้")
        break

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

    total_pills = circle_count + capsule_count + oval_count + square_count
            
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (20, 20), (200, 45), (20, 20, 20), -1)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    cv2.rectangle(annotated_frame, (20, 20), (200, 45), (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Total: {total_pills}", (40, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Real-time Pill Counter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
