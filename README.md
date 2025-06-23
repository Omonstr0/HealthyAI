# 📊 HealthyAI - Prototype V1

Bienvenue sur **HealthyAI**, une application prototype qui combine **analyse nutritionnelle automatisée** et **feedback utilisateur** pour améliorer la précision grâce au Machine Learning.

---

## 🎯 Objectif

HealthyAI a pour but d'aider les utilisateurs à :
- **Envoyer une photo de leur plat**
- **Obtenir une analyse nutritionnelle instantanée via l'API Edamam**
- **Corriger et noter les résultats** pour améliorer le modèle
- **Consulter un historique de leurs analyses**

---

## ✅ Fonctionnalités principales

### 1️⃣ **Authentification utilisateur**
- Création de compte (`signup`)
- Connexion sécurisée (`signin`)
- Réinitialisation du mot de passe (`forgot_password` + lien avec `token`)
- Déconnexion (`logout`)

> **Fichiers impliqués :**
> - `app.py` (routes)
> - `templates/signup.html`, `signin.html`, `forgot_password.html`, `reset-password.html`

---

### 2️⃣ **Upload et analyse d'image**
- L'utilisateur téléverse une photo d'un plat.
- Le modèle IA prédit le nom du plat.
- Une requête est envoyée à **Edamam API** pour récupérer les valeurs nutritionnelles pour **1 portion par défaut**.
- Affichage des calories, protéines, glucides et lipides.

> **Fichiers impliqués :**
> - `app.py` (`/upload`)
> - `predict.py` (modèle IA)
> - `dashboard.html` (formulaire et affichage)

---

### 3️⃣ **Affichage de l’image et résultats**
- Affichage de la photo téléversée directement sur le tableau de bord.
- Affichage du nom du plat détecté.
- Affichage dynamique de l'analyse nutritionnelle.

> **Fichiers impliqués :**
> - `dashboard.html`
> - `app.py` (gestion `session`)

---

### 4️⃣ **Historique personnel**
- Chaque utilisateur voit la liste de ses anciens uploads.
- Chaque image historique est affichée en miniature.
- Pour chaque image, l'utilisateur peut :
  - Revoir l'analyse (`/analyse/<upload_id>`)
  - Supprimer l'image (`/delete/<upload_id>`)

> **Fichiers impliqués :**
> - `dashboard.html`
> - `app.py`

---

### 5️⃣ **Système de feedback utilisateur**
- Pour chaque plat analysé, l'utilisateur peut :
  - Donner une note (1 à 5 étoiles)
  - Si la note est basse (≤ 3), proposer une correction manuelle du nom du plat.
- Les feedbacks sont loggés dans :
  - `feedback_log.csv`
  - Dossiers `feedback_data/<note>/` pour stocker les images par score.

> **Fichiers impliqués :**
> - `dashboard.html` (UI)
> - `app.py` (`/feedback`, `/correct/<upload_id>`)

---

### 6️⃣ **Réentraînement (prévu)**
- Le prototype prévoit :
  - La copie automatique des images corrigées dans `retraining_dataset/`
  - Le déclenchement d'un script `retrain.py` dès qu'un seuil d'images corrigées est atteint.
- Objectif : améliorer le modèle au fil du temps.

---

## 📂 Structure des données

| Dossier / Fichier | Rôle |
|-------------------|------|
| `static/uploads/` | Images uploadées |
| `feedback_data/` | Images classées par note |
| `retraining_dataset/` | Images corrigées pour le réentraînement |
| `feedback_log.csv` | Log des feedbacks |
| `database.db` | Base SQLite avec `User` et `Upload` |

---

## 🛠️ Technologies principales

- **Python 3.10+**
- **Flask** (framework backend)
- **SQLAlchemy** (ORM + SQLite)
- **PyTorch** (modèle IA)
- **Edamam Food Database API**
- **Bootstrap 5** (UI)

---

## 🚀 Déploiement (prévu)

- Prévu pour un hébergement Cloud :
  - Render.com / Railway.app pour déployer Flask facilement
  - Images possibles en local ou via un stockage Cloud (ex: AWS S3)
  - Base de données : SQLite pour prototype, Postgres conseillé pour la V2.

---

## ✅ Prochaines améliorations

- 📈 **Réentraînement automatique** en production.
- ⚖️ **Système de grammage** pour ajuster l'analyse selon le poids réel du plat.
- 🧩 **Détection multi-objets ** pour compter plusieurs items par plat.
- ☁️ **Déploiement cloud** avec stockage persistant.


**Auteur :** Équipe HealthyAI — Prototype V1 — 2025  
