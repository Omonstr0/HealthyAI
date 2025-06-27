# detect_multi_items.py
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# === Charger le modèle YOLOv8 pré-entraîné ===
model = YOLO("yolov8n.pt")  # ou yolov8m.pt pour plus de précision

# === Chemin vers l'image à tester ===
image_path = "/Users/lucas/Downloads/brocoli.jpg"  # à adapter selon ton image

# === Effectuer la détection ===
results = model(image_path)

# === Afficher les résultats visuellement ===
annotated_frame = results[0].plot()

# Affichage avec matplotlib
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Résultat YOLOv8")
plt.show()

# === Objets détectés ===
for obj in results[0].boxes.data:
    class_id = int(obj[5].item())
    confidence = obj[4].item()
    print(f"Objet : {model.names[class_id]} | Confiance : {confidence:.2f}")
