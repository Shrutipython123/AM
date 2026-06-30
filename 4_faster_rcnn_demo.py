import torch

import torchvision

import torchvision.transforms as T

from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.patches as patches

device = torch.device("cpu")

# Load pretrained Faster R-CNN

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# COCO class labels

COCO_INSTANCE_CATEGORY_NAMES = [

    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',

    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',

    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',

    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',

    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',

    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',

    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',

    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',

    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',

    'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',

    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',

    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',

    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',

    'hair drier', 'toothbrush'

]

# Transform

transform = T.Compose([T.ToTensor()])

# Same Image
image_path = "C:\\Shruti\\POC 2025\\Artificial minds\\Object_detection_demo\\datasets\\coco128\\images\\train2017\\000000000009.jpg"

img = Image.open(image_path).convert("RGB")

img_tensor = transform(img).unsqueeze(0)

# Prediction

with torch.no_grad():

    predictions = model(img_tensor)

# Visualization

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.imshow(img)

for box, label, score in zip(predictions[0]['boxes'],

                             predictions[0]['labels'],

                             predictions[0]['scores']):

    if score > 0.5:  # Confidence threshold

        xmin, ymin, xmax, ymax = box

        rect = patches.Rectangle((xmin, ymin),

                                 xmax - xmin,

                                 ymax - ymin,

                                 linewidth=2,

                                 edgecolor='red',

                                 facecolor='none')

        ax.add_patch(rect)

        ax.text(xmin, ymin,

                f"{COCO_INSTANCE_CATEGORY_NAMES[label]} {score:.2f}",

                color='yellow',

                fontsize=10,

                backgroundcolor='black')

plt.axis("off")

plt.title("Faster R-CNN Detection")

plt.show()
 