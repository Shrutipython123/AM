from ultralytics import YOLO

import cv2

# YOLOv10 nano (CPU-friendly)

model = YOLO("yolov10n.pt")

results = model(

    "https://ultralytics.com/images/bus.jpg",

    device="cpu",

    imgsz=640,

    conf=0.4

)

for r in results:

    img = r.plot()

    cv2.imshow("YOLOv10 Detection (No NMS)", img)

    cv2.waitKey(0)

cv2.destroyAllWindows()
 