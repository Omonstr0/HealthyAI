import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from datetime import datetime

# === Config ===
DATA_DIR = "retraining_dataset"
MODEL_PATH = "models/model_latest.pth"
NUM_EPOCHS = 5
BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Dataloader ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Classe vers index ===
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)
print(f"[INFO] Nouvelles classes détectées : {class_to_idx}")

# === Backup du modèle existant ===
if os.path.exists(MODEL_PATH):
    BACKUP_DIR = "models/backups"
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_name = os.path.join(BACKUP_DIR, f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    shutil.copy(MODEL_PATH, backup_name)
    print(f"[🕐] Backup créé : {backup_name}")

# === Charger modèle existant ===
model = torch.load(MODEL_PATH, map_location=DEVICE)

# === Adapter la dernière couche si le nombre de classes a changé ===
if model.fc.out_features != num_classes:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"[INFO] Couche de sortie ajustée : {num_classes} classes")

model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Fine-tuning ===
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for images, labels in tqdm(dataloader, desc=f"Époque {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[INFO] Époque {epoch+1} terminée - Loss: {total_loss:.4f}")

# === Sauvegarde du nouveau modèle
torch.save(model, MODEL_PATH)
print(f"[✅] Nouveau modèle enregistré dans {MODEL_PATH}")
