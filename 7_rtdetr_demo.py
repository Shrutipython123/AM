from ultralytics import YOLO
import cv2
# Load RT-DETR (ResNet-based)
model = YOLO("rtdetr-l.pt")
# Run inference
results = model("https://ultralytics.com/images/bus.jpg", device="cpu")
# Display results
for r in results:
   img = r.plot()
   cv2.imshow("RT-DETR Detection", img)
   cv2.waitKey(0)
cv2.destroyAllWindows()