import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ============ MODÈLE LÉGER ============
class DeepFoodCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepFoodCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)  # image 64x64

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    # ============ CONFIG ============
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BASE_DIR = os.path.dirname(__file__)
    DATASET_PATH = "dataset/dataset/images"
    MODEL_PATH = os.path.join(BASE_DIR, "models", "model_latest.pth")
    CLASS_FILE = os.path.join(os.path.dirname(__file__), "classes_food101.txt")
    STATUS_FILE = os.path.join(BASE_DIR, "training_status.json")
    IMAGE_SIZE = 64
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 1e-4

    # ============ CLASSES ============
    with open(CLASS_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    NUM_CLASSES = len(classes)

    # Vérification des dossiers
    missing_dirs = [cls for cls in classes if not os.path.isdir(os.path.join(DATASET_PATH, cls))]
    if missing_dirs:
        print(f"[ERREUR] Dossiers manquants : {missing_dirs}")
        exit(1)

    # ============ TRANSFORM ============
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ============ DATASETS ============
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # ============ ENTRAÎNEMENT ============
    model = DeepFoodCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    with open(STATUS_FILE, "w") as f:
        json.dump({"epoch": 0, "total": EPOCHS, "done": False}, f)

    for epoch in range(EPOCHS):
        model.train()
        with open(STATUS_FILE, "w") as f:
            json.dump({"epoch": epoch + 1, "total": EPOCHS, "done": False}, f)

        for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # ============ VALIDATION ============
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(val_dataset) * 100
    print(f"[✔] Accuracy validation : {accuracy:.2f}%")

    # ============ SAUVEGARDE ============
    torch.save(model.state_dict(), MODEL_PATH)
    with open(STATUS_FILE, "w") as f:
        json.dump({"epoch": EPOCHS, "total": EPOCHS, "done": True}, f)
