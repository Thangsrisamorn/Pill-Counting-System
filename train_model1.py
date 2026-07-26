from ultralytics import YOLO

MODEL_PATH = r'D:\Project\pillwebcam\runs\detect\Pill_Project\yolov8s_webcam-2\weights\best.pt'
model = YOLO(MODEL_PATH)

if __name__ == '__main__':
    results = model.train(
        data=r'D:\Project\pillwebcam\data.yaml', 
        epochs=100,                 
        patience=15,                
        imgsz=640,                  
        batch=8,                    
        workers=2,                  
        project='Pill_Project',     
        name='yolov8s_four_classes_final', 
        
        hsv_h=0.5,
        hsv_s=0.9,
        hsv_v=0.9,
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5
    )

    print(" เทรนโมเดลเสร็จ")
