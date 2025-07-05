import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ============ MODÈLE ============
class DeepFoodCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepFoodCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    # ============ CONFIG ============
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Entraînement sur : {DEVICE}")

    DATASET_PATH = "dataset/dataset/images"
    MODEL_PATH = "models/model_latest.pth"
    CLASS_FILE = "classes_food101.txt"
    STATUS_FILE = "training_status.json"
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 75
    LR = 1e-4

    # ============ CLASSES ============
    with open(CLASS_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    NUM_CLASSES = len(classes)

    # Vérification 1 : dossiers manquants
    missing_dirs = [cls for cls in classes if not os.path.isdir(os.path.join(DATASET_PATH, cls))]
    if missing_dirs:
        print(f"[ERREUR] Les dossiers suivants sont manquants dans {DATASET_PATH} : {missing_dirs}")
        print("💡 Vérifie que chaque classe de classes_food101.txt a un dossier correspondant avec des images.")
        exit(1)

    # Vérification 2 : dossiers non référencés
    existing_dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    unused_dirs = [d for d in existing_dirs if d not in classes]
    if unused_dirs:
        print(f"[AVERTISSEMENT] Les dossiers suivants sont présents dans {DATASET_PATH} mais non référencés dans {CLASS_FILE} : {unused_dirs}")
        print("💡 Si ce sont de nouvelles classes, ajoute-les à classes_food101.txt.")

    # ============ TRANSFORMATIONS ============
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ============ DATASETS ============
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ============ ENTRAÎNEMENT ============
    model = DeepFoodCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # ✅ Met à jour le fichier JSON de statut
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
        print(f"-> Loss: {running_loss/len(train_loader):.4f}")
        scheduler.step()

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
    print(f"[Validation] Accuracy: {accuracy:.2f}%")

    # ============ SAUVEGARDE ============
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[✔] Modèle sauvegardé dans {MODEL_PATH}")

    # ✅ Entraînement terminé : mise à jour du JSON
    with open(STATUS_FILE, "w") as f:
        json.dump({"epoch": EPOCHS, "total": EPOCHS, "done": True}, f)
