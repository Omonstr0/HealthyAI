import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from datetime import datetime

# === Import du modèle défini dans train_from_scratch.py ===
from train_from_scratch import DeepFoodCNN

# === Config ===
DATA_DIR = "retraining_dataset"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_latest.pth")
NUM_EPOCHS = 5
BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Chemin absolu du modèle : {MODEL_PATH}")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # même taille que l'entraînement initial
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # même normalisation
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
    BACKUP_DIR = os.path.join(BASE_DIR, "models", "backups")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_name = os.path.join(BACKUP_DIR, f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    shutil.copy(MODEL_PATH, backup_name)
    print(f"[🕐] Backup créé : {backup_name}")

# === Création du modèle
model = DeepFoodCNN(num_classes).to(DEVICE)

# === Chargement des poids existants
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("[INFO] Poids chargés depuis le fichier .pth")

# === Entraînement (fine-tuning)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

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

# === Sauvegarde des nouveaux poids
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"[✅] Nouveau modèle sauvegardé dans {MODEL_PATH}")
