from ultralytics import YOLO
import cv2
# Load YOLOv8 nano (CPU-friendly)
model = YOLO("yolov8n.pt")
# Run inference on sample COCO image
results = model("https://ultralytics.com/images/bus.jpg", device="cpu")
# Display results
for r in results:
   img = r.plot()
   cv2.imshow("YOLOv8 Detection", img)
   cv2.waitKey(0)
cv2.destroyAllWindows()