# 📊 HealthyAI - Prototype V1

Bienvenue sur **HealthyAI**, une application prototype qui combine **analyse nutritionnelle automatisée** et **feedback utilisateur** pour améliorer la précision grâce au Machine Learning.

dataset : https://www.kaggle.com/datasets/dansbecker/food-101?resource=download

---

## 🎯 Objectif

HealthyAI a pour but d'aider les utilisateurs à :
- **Envoyer une photo de leur plat**
- **Obtenir une analyse nutritionnelle instantanée**
- **Corriger et évaluer les résultats** pour améliorer le modèle
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
- On récupére les valeurs nutritionnelles pour **1 portion par défaut** via le fichier "plats.csv".
- Affichage des calories, protéines, glucides et lipides.

> **Fichiers impliqués :**
> - `app.py` (`/upload`)
> - `predict.py` (modèle IA)
> - `dashboard.html` (formulaire et affichage)
> - `plat.csv` (fichier csv)

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
  - Evalutation de la prédiction
  - Si l'évaluation est négative, proposer une correction manuelle du nom du plat.
- Les feedbacks sont loggés dans :
  - `feedback_log.csv`

---

### 6️⃣ **Réentraînement**
- Le prototype :
  - La copie automatique des images corrigées.
  - Le déclenchement d'un script `train_from_scratch.py` dès qu'un seuil d'images corrigées est atteint.
- Objectif : améliorer le modèle au fil du temps.

---

## 📂 Structure des données

| Dossier / Fichier | Rôle |
|-------------------|------|
| `static/uploads/` | Images uploadées |
| `feedback_data/` | Images classées |
| `feedback_log.csv` | Log des feedbacks |
| `database.db` | Base PostgreSQL avec `User`, `Upload` et `UserProfile` |

---

## 🛠️ Technologies principales

- **Python 3.10+**
- **Flask** (framework backend)
- **PyTorch** (modèle IA)
- **Bootstrap 5** (UI)

---

## 🚀 Déploiement

- Hébergement Cloud :
  - Render.com / Railway.app pour déployer Flask facilement
  - Dataset déposé sur Hugging Face
  - Base de données : PostgreSQL

---

## ✅ Prochaines améliorations

- 📈 **Intégration API** nutritionnelle pour des données riches
- ⚖️ Déploiement vers un modèle **multi items**.
- 🧩 Développement d’une **application mobile**.
- ☁️ Intégration d'une dimension **santé personnalisée**.


**Auteur :** Équipe HealthyAI — Prototype V1 — 2025  
