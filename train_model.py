rom ultralytics import YOLO

model = YOLO('yolov8s.pt')

if __name__ == '__main__':
    results = model.train(
        data='data.yaml',           
        epochs=50,                 
        imgsz=640,               
        batch=8,                  
        workers=2,              
        project='Pill_Project',   
        name='yolov8s_webcam',      

        hsv_h=0.5,      # สุ่มเปลี่ยนเฉดสี (Hue)
        hsv_s=0.9,      # สุ่มเปลี่ยนความสดของสี (Saturation) ทำให้มีทั้งภาพสีสดและสีซีด
        hsv_v=0.9,      # สุ่มเปลี่ยนความสว่าง (Value) ป้องกันปัญหาแสงจาก Webcam
        degrees=180.0,  # สุ่มหมุนภาพสูงสุด 180 องศา (สำหรับเม็ดยาที่ตกแบบสุ่มมุม)
        fliplr=0.5,     # สุ่มพลิกซ้ายขวา 50%
        flipud=0.5      # สุ่มพลิกบนล่าง 50%
    )
    
    print("เทรนโมเดลเสร็จ")
