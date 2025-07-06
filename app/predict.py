import os
import torch
from torchvision import transforms
from PIL import Image
from train_from_scratch import DeepFoodCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chargement des classes
CLASS_FILE = os.path.join(BASE_DIR, "classes_food101.txt")
with open(CLASS_FILE, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

num_classes = len(class_labels)

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Résolution dynamique du chemin vers le modèle
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_latest.pth")

# Chargement du modèle
model = DeepFoodCNN(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_food(image_path):
    """
    Prend le chemin d'une image, retourne un label prédit
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # batch de taille 1

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_labels[predicted.item()]

    return predicted_label
