from ultralytics import YOLO

# 1. โหลดโมเดลเดิม (เพื่อให้มันจำความรู้เดิม 2 คลาสแรกได้)
# เช็กให้ชัวร์ว่า Path นี้ถูกต้องตามที่อยู่ไฟล์ในเครื่องของคุณนะครับ
MODEL_PATH = r'D:\Project\pillwebcam\runs\detect\Pill_Project\yolov8s_webcam-2\weights\best.pt'
model = YOLO(MODEL_PATH)

if __name__ == '__main__':
    print("🚀 กำลังเริ่มเทรนโมเดลเพื่อเพิ่มคลาส Oval และ Square...")
    
    # 2. เริ่มเทรนด้วยข้อมูลชุดใหม่ผสมชุดเก่า
    # ใส่ Path เต็มของ data.yaml เพื่อป้องกันข้อผิดพลาดในการหาไฟล์
    results = model.train(
        data=r'D:\Project\pillwebcam\data.yaml', 
        epochs=100,                 
        patience=15,                
        imgsz=640,                  
        batch=8,                    
        workers=2,                  
        project='Pill_Project',     
        name='yolov8s_four_classes_final', # ตั้งชื่อใหม่ให้ชัดเจน
        
        # ตั้งค่า Data Augmentation 
        hsv_h=0.5,
        hsv_s=0.9,
        hsv_v=0.9,
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5
    )
    
    print("🎉 เทรนโมเดลเสร็จสมบูรณ์! ได้ไฟล์ best.pt ใหม่ในโฟลเดอร์ Pill_Project แล้วครับ")