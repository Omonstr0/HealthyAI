import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============ MODÈLE IDENTIQUE AU TRAIN PRINCIPAL ============
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
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_PATH = "models/model_72pct.pth"
    RETRAIN_PATH = "retraining_dataset"
    CLASS_FILE = "classes_food101.txt"
    IMAGE_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4

    # Labels et classes
    with open(CLASS_FILE) as f:
        class_names = [line.strip() for line in f]
    NUM_CLASSES = len(class_names)

    # Transforms simples (on ne veut pas trop de bruit ici)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Dataset de corrections utilisateur
    retrain_dataset = datasets.ImageFolder(RETRAIN_PATH, transform=transform)
    retrain_loader = DataLoader(retrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Modèle à partir du fichier existant
    model = DeepFoodCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Optimiseur + perte + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Réentraînement
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for imgs, labels in tqdm(retrain_loader, desc=f"[Retrain Epoch {epoch+1}/{EPOCHS}]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"-> Loss: {running_loss/len(retrain_loader):.4f}")
        scheduler.step()

    # Sauvegarde finale
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[✔] Modèle réentraîné sauvegardé dans {MODEL_PATH}")
