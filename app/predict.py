# predict.py

import torch
from torchvision import transforms
from PIL import Image
from train_from_scratch import DeepFoodCNN  # Assure-toi que ce nom correspond
import os

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Chargement des classes
with open("classes_food101.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

num_classes = len(class_labels)

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Chargement du modèle
model = DeepFoodCNN(num_classes=num_classes)
model.load_state_dict(torch.load("../models/model_latest.pth", map_location=DEVICE))
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
