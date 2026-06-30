import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import urllib.request

# -----------------------------------------------------------------------------
# 1) ENVIRONMENT / DEVICE SETUP
# -----------------------------------------------------------------------------
# Choose where computations run.
# "cpu" keeps this demo simple and compatible on most machines.
device = torch.device("cpu")

# -----------------------------------------------------------------------------
# 2) LOAD PRETRAINED MODEL
# -----------------------------------------------------------------------------
# ResNet18 is a well-known image classification model.
# pretrained=True means it already learned from ImageNet (millions of images),
# so we can use it directly without training from scratch.
model = models.resnet18(pretrained=True)

# model.eval() puts the model in evaluation mode (important for inference).
# This disables training-specific behaviors and makes predictions consistent.
model.eval()

# -----------------------------------------------------------------------------
# 3) LABEL NAMES (CLASS DICTIONARY)
# -----------------------------------------------------------------------------
# Model outputs class indices (numbers). We download label names so results are
# human-readable (for example: "sports car", "dog", "bus", etc.).
# This file is downloaded once and reused.
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")
with open("imagenet_classes.txt") as f:
   categories = [line.strip() for line in f.readlines()]

# -----------------------------------------------------------------------------
# 4) PREPROCESSING PIPELINE (VERY IMPORTANT)
# -----------------------------------------------------------------------------
# This is where preprocessing happens before prediction:
# - Resize to 224x224 (ResNet18 expected input size)
# - Convert image to tensor
# - Normalize channels using ImageNet mean/std so input distribution matches
#   what the pretrained model saw during training.
transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
])

# -----------------------------------------------------------------------------
# 5) INPUT IMAGE SELECTION + PREP
# -----------------------------------------------------------------------------
# Use one fixed image for consistent comparison across different model demos.
image_path = "C:\\Shruti\\POC 2025\\Artificial minds\\Object_detection_demo\\datasets\\coco128\\images\\train2017\\000000000009.jpg"

# Open image and ensure RGB format (3 channels).
img = Image.open(image_path).convert("RGB")

# Apply preprocessing and add batch dimension.
# unsqueeze(0) changes shape from [C,H,W] to [1,C,H,W] because the model
# expects a batch, even if there is only one image.
input_tensor = transform(img).unsqueeze(0)

# -----------------------------------------------------------------------------
# 6) INFERENCE (PREDICTION STAGE)
# -----------------------------------------------------------------------------
# This is where prediction happens.
# torch.no_grad() disables gradient calculations because we are not training,
# which makes inference faster and memory-efficient.
with torch.no_grad():
   output = model(input_tensor)

# Convert raw model scores (logits) into probabilities using softmax.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# -----------------------------------------------------------------------------
# 7) TOP-5 RESULT EXTRACTION + TEXT OUTPUT
# -----------------------------------------------------------------------------
# Get the 5 most likely classes and their confidence scores.
top5_prob, top5_catid = torch.topk(probabilities, 5)
top1_label = categories[top5_catid[0]]
top1_conf = top5_prob[0].item() * 100

print("\n===== Prediction Summary =====")
print("Model: ResNet18")
print(f"Top-1: {top1_label} ({top1_conf:.2f}%)")
print("Top-5:")
for rank in range(top5_prob.size(0)):
   label = categories[top5_catid[rank]]
   conf = top5_prob[rank].item() * 100
   print(f"{rank+1}. {label}: {conf:.2f}%")

# -----------------------------------------------------------------------------
# 8) VISUAL PRESENTATION
# -----------------------------------------------------------------------------
# Show the original image with the top predicted class and confidence.
# This makes the output easier to explain to a non-technical audience.
plt.imshow(img)
plt.axis("off")
plt.title(f"Top Prediction:\n{top1_label} ({top1_conf:.2f}%)")
plt.show()