import os
import urllib.request

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


device = torch.device("cpu")

image_path = "C:\\Shruti\\POC 2025\\Artificial minds\\Object_detection_demo\\datasets\\coco128\\images\\train2017\\000000000009.jpg"
labels_file = "imagenet_classes.txt"
mnist_checkpoint = "mnist_cnn.pth"


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def ensure_imagenet_labels(path):
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, path)

    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def train_or_load_mnist_cnn(checkpoint_path):
    model = SimpleCNN().to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training MNIST CNN checkpoint (one-time)...")
    for epoch in range(2):
        running_loss = 0.0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"MNIST Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    print(f"Saved MNIST CNN checkpoint: {checkpoint_path}")
    return model


def imagenet_top5(model, img, categories):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)

    top5 = []
    for rank in range(top5_prob.size(0)):
        top5.append((categories[top5_catid[rank].item()], top5_prob[rank].item() * 100))

    return top5


def mnist_top5(model, img):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)

    top5 = []
    for rank in range(top5_prob.size(0)):
        top5.append((str(top5_catid[rank].item()), top5_prob[rank].item() * 100))

    return top5


def print_summary(model_name, top5):
    top1_label, top1_conf = top5[0]
    print("\n===== Prediction Summary =====")
    print(f"Model: {model_name}")
    print(f"Top-1: {top1_label} ({top1_conf:.2f}%)")
    print("Top-5:")
    for rank, (label, conf) in enumerate(top5, start=1):
        print(f"{rank}. {label}: {conf:.2f}%")


def main():
    categories = ensure_imagenet_labels(labels_file)
    img = Image.open(image_path).convert("RGB")

    resnet = models.resnet18(pretrained=True).to(device)
    mobilenet = models.mobilenet_v2(pretrained=True).to(device)
    efficientnet = models.efficientnet_b0(pretrained=True).to(device)

    resnet.eval()
    mobilenet.eval()
    efficientnet.eval()

    mnist_cnn = train_or_load_mnist_cnn(mnist_checkpoint)

    cnn_top5 = mnist_top5(mnist_cnn, img)
    resnet_top5 = imagenet_top5(resnet, img, categories)
    mobilenet_top5 = imagenet_top5(mobilenet, img, categories)
    efficientnet_top5 = imagenet_top5(efficientnet, img, categories)

    print_summary("CNN (MNIST digits)", cnn_top5)
    print_summary("ResNet18", resnet_top5)
    print_summary("MobileNetV2", mobilenet_top5)
    print_summary("EfficientNet-B0", efficientnet_top5)

    cnn_label, cnn_conf = cnn_top5[0]
    resnet_label, resnet_conf = resnet_top5[0]
    mobilenet_label, mobilenet_conf = mobilenet_top5[0]
    efficientnet_label, efficientnet_conf = efficientnet_top5[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "Top-1 Comparison\n"
        f"CNN(MNIST): {cnn_label} ({cnn_conf:.1f}%) | "
        f"ResNet18: {resnet_label} ({resnet_conf:.1f}%)\n"
        f"MobileNetV2: {mobilenet_label} ({mobilenet_conf:.1f}%) | "
        f"EfficientNet-B0: {efficientnet_label} ({efficientnet_conf:.1f}%)"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
