from ultralytics import YOLO
# Load a lightweight YOLO model
model = YOLO("yolov8n.pt")
# This command automatically downloads the COCO128
model.val(data="coco128.yaml", device="cpu")