import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ==== MODELE LIGHT ====
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # ==== CONFIG ====
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Entraînement sur : {DEVICE}")

    DATASET_PATH = "dataset/images"

    BASE_DIR = os.path.dirname(__file__)
    CLASS_FILE = os.path.join(BASE_DIR, "classes_food5.txt")
    MODEL_PATH = os.path.join(BASE_DIR, "models/model_small.pth")
    STATUS_FILE = os.path.join(BASE_DIR, "training_status.json")
    IMAGE_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4

    # ==== CHARGEMENT DES CLASSES ====
    with open(CLASS_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    NUM_CLASSES = len(classes)

    # ==== VERIFICATION ====
    missing_dirs = [cls for cls in classes if not os.path.isdir(os.path.join(DATASET_PATH, cls))]
    if missing_dirs:
        print(f"[ERREUR] Dossiers manquants : {missing_dirs}")
        exit(1)

    # ==== AUGMENTATION PUISSANTE POUR PETIT DATASET ====
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ==== CHARGEMENT DU DATASET ====
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ==== MODELE / LOSS / OPTIM ====
    model = SmallCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ==== ENTRAINEMENT ====
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Mise à jour du JSON
        with open(STATUS_FILE, "w") as f:
            json.dump({"epoch": epoch + 1, "total": EPOCHS}, f)

        for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"-> Loss: {running_loss / len(train_loader):.4f}")

    # ==== VALIDATION ====
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(val_dataset) * 100
    print(f"[Validation] Accuracy: {accuracy:.2f}%")

    # ==== SAUVEGARDE ====
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[✔] Modèle sauvegardé dans {MODEL_PATH}")

    # JSON terminé
    with open(STATUS_FILE, "w") as f:
        json.dump({"epoch": EPOCHS, "total": EPOCHS, "done": True}, f)
