import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) ENVIRONMENT / DEVICE SETUP
# -----------------------------------------------------------------------------
# We choose where the model calculations happen.
# "cpu" means the code runs on the processor (works on all systems and is fine
# for this educational MNIST demo).
device = torch.device("cpu")

# -----------------------------------------------------------------------------
# 2) PREPROCESSING
# -----------------------------------------------------------------------------
# This is where preprocessing happens.
# transforms.ToTensor() converts each image from pixel format [0..255]
# to a tensor with normalized float values [0..1], which the neural network
# can understand and process.
transform = transforms.ToTensor()

# -----------------------------------------------------------------------------
# 3) DATASET LOADING (TRAIN + TEST)
# -----------------------------------------------------------------------------
# Training set: used for learning model weights.
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

# Test set: kept separate and used only to measure final performance.
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

# DataLoader creates mini-batches for efficient processing.
# - batch_size=64: model sees 64 images at a time.
# - shuffle=True (train): random order improves learning quality.
# - shuffle=False (test): consistent order for stable evaluation.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# -----------------------------------------------------------------------------
# 4) MODEL DEFINITION (CNN)
# -----------------------------------------------------------------------------
# CNN = Convolutional Neural Network, designed for image understanding.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolution: learns simple visual patterns (edges/strokes).
        self.conv1 = nn.Conv2d(1, 16, 3)

        # Second convolution: learns deeper/more complex patterns.
        self.conv2 = nn.Conv2d(16, 32, 3)

        # Pooling reduces spatial size (faster + helps generalization).
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers convert learned features into class decision.
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes = digits 0 to 9

    def forward(self, x):
        # Forward pass: defines the path from input image to output scores.
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)   # flatten feature maps into vector
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)               # raw scores (logits) for 10 classes
        return x


# Build model object and place it on chosen device.
model = SimpleCNN().to(device)

# Loss function: compares model prediction vs true label.
criterion = nn.CrossEntropyLoss()

# Optimizer: updates model weights to minimize loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -----------------------------------------------------------------------------
# 5) TRAINING LOOP (LEARNING STAGE)
# -----------------------------------------------------------------------------
# This is where training happens.
# Each epoch = one complete pass over all training images.
for epoch in range(2):  # 2 epochs kept short for demo
    running_loss = 0.0

    for images, labels in trainloader:
        # Step 1: clear old gradients from previous batch.
        optimizer.zero_grad()

        # Step 2: forward pass (model predicts output scores).
        outputs = model(images)

        # Step 3: compute prediction error.
        loss = criterion(outputs, labels)

        # Step 4: backward pass (compute gradients).
        loss.backward()

        # Step 5: optimizer updates model parameters.
        optimizer.step()

        running_loss += loss.item()

    # End-of-epoch progress report.
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

print("Training Completed!")


# -----------------------------------------------------------------------------
# 6) EVALUATION ON TEST DATA (NO LEARNING HERE)
# -----------------------------------------------------------------------------
# This is where model performance is measured on unseen data.
correct = 0
total = 0

# torch.no_grad() disables gradient computation (faster and memory-efficient).
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


# -----------------------------------------------------------------------------
# 7) VISUAL DEMO FOR PRESENTATION
# -----------------------------------------------------------------------------
# Show 5 test images with actual digit vs predicted digit.
# This makes results easy to explain to a non-technical audience.
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i in range(5):
    img, label = testset[i]

    # Add batch dimension because model expects batches, not single image tensor.
    img_input = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img_input)
        _, predicted = torch.max(output, 1)

    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].set_title(f"Actual: {label}\nPred: {predicted.item()}")
    axes[i].axis("off")

plt.show()