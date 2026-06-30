from ultralytics import YOLO, RTDETR
import time
image = "https://ultralytics.com/images/bus.jpg"
# YOLOv8
yolo = YOLO("yolov8n.pt")
start = time.time()
yolo(image, device="cpu")
yolo_time = time.time() - start
# RT-DETR
rtdetr = RTDETR("rtdetr-r50.pt")
start = time.time()
rtdetr(image, device="cpu")
rtdetr_time = time.time() - start
print(f"YOLOv8 inference time (CPU): {yolo_time:.2f}s")
print(f"RT-DETR inference time (CPU): {rtdetr_time:.2f}s")