import os
import torch
from torchvision import transforms
from PIL import Image
from train_from_scratch import SmallCNN

# ==== CONFIGURATION ====
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==== CLASSES ====
CLASS_FILE = os.path.join(BASE_DIR, "classes_food5.txt")
with open(CLASS_FILE, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

NUM_CLASSES = len(class_labels)

# ==== PRETRAITEMENT ====
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # cohérent avec l'entrée du modèle
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==== CHARGEMENT DU MODELE ====
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_small.pth")
model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== PREDICTION ====
def predict_food(image_path):
    """
    Prend le chemin d'une image, retourne un label prédit
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_labels[predicted.item()]

    return predicted_label
