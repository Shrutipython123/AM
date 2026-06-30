import torch

import torchvision.models as models

import torchvision.transforms as transforms

from PIL import Image

import matplotlib.pyplot as plt

import urllib.request

device = torch.device("cpu")

# Load Pretrained MobileNetV2

model = models.mobilenet_v2(pretrained=True)

model.eval()

# Download ImageNet class labels (first time only)

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

urllib.request.urlretrieve(url, "imagenet_classes.txt")

with open("imagenet_classes.txt") as f:

    categories = [line.strip() for line in f.readlines()]

# Transform

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

# Same Image

image_path = "C:\\Shruti\\POC 2025\\Artificial minds\\Object_detection_demo\\datasets\\coco128\\images\\train2017\\000000000009.jpg"

img = Image.open(image_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0)

# Prediction

with torch.no_grad():

    output = model(input_tensor)

probabilities = torch.nn.functional.softmax(output[0], dim=0)

top5_prob, top5_catid = torch.topk(probabilities, 5)

top1_label = categories[top5_catid[0]]
top1_conf = top5_prob[0].item() * 100

print("\n===== Prediction Summary =====")
print("Model: MobileNetV2")
print(f"Top-1: {top1_label} ({top1_conf:.2f}%)")
print("Top-5:")
for rank in range(top5_prob.size(0)):
    label = categories[top5_catid[rank]]
    conf = top5_prob[rank].item() * 100
    print(f"{rank+1}. {label}: {conf:.2f}%")

# Visualize

plt.imshow(img)

plt.axis("off")

plt.title(f"Top Prediction:\n{top1_label} ({top1_conf:.2f}%)")

plt.show()
 