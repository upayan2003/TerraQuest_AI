import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm

# -------------------- Model --------------------

class TerrainCNN(nn.Module):
    def __init__(self, num_classes, input_type='ground'):
        super(TerrainCNN, self).__init__()
        self.input_type = input_type

        if input_type == 'satellite':
            # Lightweight model with Adaptive Pooling
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),   # [B, 128, 1, 1]
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )
            
        elif input_type == 'ground':
            # Spatial-preserving model with full flatten
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # [B, 32, 128, 128]
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # [B, 64, 64, 64]
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # [B, 128, 32, 32]
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 32 * 32, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    def forward(self, x):
        if self.input_type == 'satellite':
            return self.net(x)
        elif self.input_type == 'ground':
            x = self.features(x)
            return self.classifier(x)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

# -------------------- Augmentation --------------------

def get_transforms(image_size=(256, 256)):
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    }

# -------------------- Dataset Loader --------------------

def get_dataloaders(data_dir, batch_size=32, image_size=(256, 256)):
    transforms_dict = get_transforms(image_size)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transforms_dict["train"])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transforms_dict["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names

# -------------------- Training Loop --------------------

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=0.0005, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=total_loss / total, acc=correct / total)

        val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# -------------------- Evaluation --------------------

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# -------------------- Prediction --------------------

def predict_image(img_path, model_path, class_names, input_type, image_size=(256, 256), device='cpu', threshold_diff=40.0):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    model = TerrainCNN(num_classes=len(class_names), input_type=input_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Get top 2 predictions
    top_probs, top_indices = torch.topk(probabilities, 2)
    top1_conf = top_probs[0].item() * 100
    top2_conf = top_probs[1].item() * 100
    top1_class = class_names[top_indices[0].item()]
    top2_class = class_names[top_indices[1].item()]

    confidence_scores = {class_names[i]: f"{(probabilities[i].item() * 100):.2f}%" for i in range(len(class_names))}

    if (top1_conf - top2_conf) <= threshold_diff:
        prediction = f"{top1_class}&{top2_class}"
    else:
        prediction = top1_class

    return prediction, confidence_scores
