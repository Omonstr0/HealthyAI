from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from predict import model, transform, class_labels, DEVICE
from datetime import datetime
import os
import secrets
import uuid
import subprocess
import shutil
import csv
import datetime
import torch
import pandas as pd
import json

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
if not app.config["SQLALCHEMY_DATABASE_URI"]:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Détection de l’environnement (Render ou local)
if os.environ.get("RENDER") == "true":
    UPLOAD_FOLDER = '/mnt/uploads'
    FEEDBACK_CSV_PATH = "feedback_cloud_csv/feedback_log.csv"
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    FEEDBACK_CSV_PATH = os.path.join(app.root_path, "feedback_csv", "feedback_log.csv")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(FEEDBACK_CSV_PATH), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========  Auto-download du dataset si vide ==========
DATASET_DIR = "dataset/images/"
ZIP_URL = "https://huggingface.co/datasets/Omonstr0/dataset-healthyai/resolve/main/dataset.zip"
ZIP_PATH = "dataset.zip"

if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) < 10:
    import urllib.request, zipfile
    print("[INFO] Téléchargement du dataset...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("[INFO] Dataset prêt.")


db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Injection globale de la date/heure dans tous les templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow()}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    reset_token = db.Column(db.String(256), nullable=True)

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), db.ForeignKey('user.email'), nullable=False)
    age = db.Column(db.Integer)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    sex = db.Column(db.String(10))
    activity = db.Column(db.Float)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user = db.relationship('User', backref=db.backref('uploads', lazy=True))
    dish_name = db.Column(db.String(150), nullable=True)
    rating = db.Column(db.Integer)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

nutrition_df = pd.read_csv(os.path.join(app.root_path, 'plats.csv'))

def get_nutrition_from_food(dish_name):
    csv_dish_name = dish_name.lower().replace(" ", "_")
    row = nutrition_df[nutrition_df['name'].str.lower() == csv_dish_name]

    if row.empty:
        row = nutrition_df[nutrition_df['name_fr'].str.lower() == csv_dish_name]

    if not row.empty:
        return {
            "kcal": float(row["kcal"].values[0]),
            "protein_g": float(row["protein_g"].values[0]),
            "carbs_g": float(row["carbs_g"].values[0]),
            "fat_g": float(row["fat_g"].values[0]),
            "name_fr": row["name_fr"].values[0]  # affichage propre
        }
    else:
        # Aucune correspondance trouvée
        return {
            "kcal": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "name_fr": dish_name.replace("_", " ")  # garde le nom personnalisé
        }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_email'] = user.email
            flash('Connexion réussie !', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Email ou mot de passe incorrect.', 'danger')
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']
        if password != confirm:
            flash('Les mots de passe ne correspondent pas.', 'warning')
            return redirect(url_for('signup'))
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Un compte existe déjà avec cet email.', 'warning')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Compte créé avec succès. Vous pouvez vous connecter.', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            token = secrets.token_urlsafe(32)
            user.reset_token = token
            db.session.commit()
            reset_link = url_for('reset_password', token=token, _external=True)
            print(f"[DEV] Lien de réinitialisation : {reset_link}")
            flash('Un email de réinitialisation a été envoyé (vérifiez la console pour le lien)', 'info')
        else:
            flash('Aucun compte trouvé avec cet email.', 'warning')
        return redirect(url_for('signin'))
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user:
        flash('Lien invalide ou expiré.', 'danger')
        return redirect(url_for('signin'))
    if request.method == 'POST':
        password = request.form['password']
        confirm = request.form['confirm']
        if password != confirm:
            flash('Les mots de passe ne correspondent pas.', 'warning')
            return redirect(request.url)
        user.password = generate_password_hash(password, method='pbkdf2:sha256')
        user.reset_token = None
        db.session.commit()
        flash('Mot de passe mis à jour avec succès !', 'success')
        return redirect(url_for('signin'))
    return render_template('reset-password.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Veuillez vous connecter pour accéder au tableau de bord.', 'danger')
        return redirect(url_for('signin'))

    uploads = Upload.query.filter_by(user_id=session['user_id']).order_by(Upload.timestamp.desc()).all()

    # === Lecture du fichier training_status.json ===
    training_status = None
    status_path = os.path.join(app.root_path, "training_status.json")

    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                status_data = json.load(f)
                # On garde le status uniquement si l'entraînement est en cours
                if not status_data.get("done", True):
                    training_status = status_data
        except Exception:
            training_status = None

    # === Variables d'affichage
    show_progress = False
    percent = 0
    status_label = ""

    if training_status:
        show_progress = True
        percent = int(100 * training_status.get("epoch", 0) / training_status.get("total", 1))
        status_label = "En cours"

    return render_template(
        'dashboard.html',
        uploads=uploads,
        nutrition=session.get('last_nutrition'),
        image_url=session.get('last_image'),
        detected_dish=session.get('last_dish'),
        corrected_id=session.pop('corrected_id', None),
        show_progress=show_progress,
        percent=percent,
        status_label=status_label
    )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:
        flash('Veuillez vous connecter.', 'danger')
        return redirect(url_for('signin'))

    file = request.files.get('photo')
    if not file or file.filename == '':
        flash('Aucune image sélectionnée.', 'warning')
        return redirect(url_for('dashboard'))

    if allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        static_upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(static_upload_folder, exist_ok=True)
        dst_path = os.path.join(static_upload_folder, filename)

        # Évite la SameFileError en local
        if os.path.abspath(filepath) != os.path.abspath(dst_path):
            shutil.copy(filepath, dst_path)

        # Vérification de compatibilité
        try:
            with Image.open(filepath) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            os.remove(filepath)
            flash("Image non compatible. Veuillez choisir un fichier image valide (JPG, PNG).", "danger")
            return redirect(url_for('dashboard'))

        # Prédiction + confiance
        img = Image.open(filepath).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        model.eval()

        if DEVICE.type == 'mps':
            img_tensor = img_tensor.to(torch.float32)
            model.to("mps")
            img_tensor = img_tensor.to("mps")
        else:
            model.to("cpu")
            img_tensor = img_tensor.to("cpu")

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence = torch.max(probs).item()
            predicted_idx = torch.argmax(probs, dim=1).item()
            predicted_label = class_labels[predicted_idx]

        label_formatted = predicted_label.replace("_", " ").title()

        # nutrition
        nutrition = get_nutrition_from_food(label_formatted)

        # Enregistrement DB
        new_upload = Upload(
            filename=filename,
            user_id=session['user_id'],
            dish_name=nutrition.get("name_fr", label_formatted)
        )
        db.session.add(new_upload)
        db.session.commit()

        # Session
        session['last_nutrition'] = nutrition
        session['last_dish'] = label_formatted
        session['last_confidence'] = round(confidence, 4)

        # Historique
        uploads = Upload.query.filter_by(user_id=session['user_id']).order_by(Upload.timestamp.desc()).all()

        # Pré-écriture du log feedback (sans rating pour l’instant)
        log_path = FEEDBACK_CSV_PATH
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "plat_prédit", "note", "correction", "date", "confiance", "re-entrainement"])

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                filename, label_formatted, "", "", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(confidence, 4), ""
            ])

        session['last_image'] = url_for('static', filename=f'uploads/{filename}')

        return render_template('dashboard.html',
                               uploads=uploads,
                               nutrition=nutrition,
                               detected_dish=label_formatted,
                               image_url=session['last_image'])

    else:
        flash('Format de fichier non autorisé. Veuillez choisir un JPG ou PNG.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/analyse/<int:upload_id>')
def analyse(upload_id):
    upload = Upload.query.get_or_404(upload_id)

    if upload.user_id != session.get('user_id'):
        flash("Accès non autorisé.", "danger")
        return redirect(url_for('dashboard'))

    # Met à jour les infos pour affichage
    label = upload.dish_name
    nutrition = get_nutrition_from_food(label)

    session['last_nutrition'] = nutrition
    session['last_image'] = url_for('static', filename=f'uploads/{upload.filename}')
    session['last_dish'] = label

    # Réinitialise le rating à None après analyse (si correction déjà faite)
    if upload.rating == 1:
        upload.rating = None
        db.session.commit()

    return redirect(url_for('dashboard'))


@app.route("/calorie", methods=["GET", "POST"])
def calorie():
    if 'user_id' not in session:
        flash("Veuillez vous connecter pour accéder à cette page.", 'warning')
        return redirect(url_for('signin'))

    calories = None
    protein_low = None
    protein_high = None
    edit_mode = request.args.get("edit") == "1"

    user_email = session.get("user_email")
    profile = UserProfile.query.filter_by(user_email=user_email).first()

    if request.method == "POST":
        try:
            if edit_mode:
                # Si le formulaire d’édition est activé
                age = int(request.form.get("age"))
                height = float(request.form.get("height"))
                weight = float(request.form.get("weight"))
                sex = request.form.get("sex")
                activity = float(request.form.get("activity"))

                if profile:
                    profile.age = age
                    profile.height = height
                    profile.weight = weight
                    profile.sex = sex
                    profile.activity = activity
                else:
                    profile = UserProfile(
                        user_email=user_email,
                        age=age,
                        height=height,
                        weight=weight,
                        sex=sex,
                        activity=activity
                    )
                    db.session.add(profile)

                db.session.commit()
                flash("Informations enregistrées avec succès", "success")
                return redirect(url_for('calorie'))

            elif profile:
                # Pas en mode édition : utiliser les données existantes
                age = profile.age
                height = profile.height
                weight = profile.weight
                sex = profile.sex
                activity = profile.activity

            else:
                flash("Aucune donnée disponible. Veuillez remplir le formulaire d'abord.", "danger")
                return redirect(url_for('calorie'))

            # Calcul nutritionnel dans les deux cas
            bmr = 10 * weight + 6.25 * height - 5 * age + (5 if sex == "male" else -161)
            calories = round(bmr * activity)
            protein_low = round(weight * 0.8)
            protein_high = round(weight * 1.0)

        except Exception as e:
            print("Erreur :", e)
            flash("Erreur dans les données envoyées. Veuillez réessayer.", "danger")

    return render_template("calorie.html",
                           profile=profile,
                           calories=calories,
                           protein_low=protein_low,
                           protein_high=protein_high,
                           edit_mode=edit_mode)


@app.route('/delete/<int:upload_id>', methods=['POST'])
def delete_upload(upload_id):
    if 'user_id' not in session:
        flash("Accès non autorisé", 'danger')
        return redirect(url_for('signin'))

    upload = Upload.query.get_or_404(upload_id)

    # Vérifier que l'utilisateur est propriétaire
    if upload.user_id != session['user_id']:
        flash("Action non autorisée.", 'danger')
        return redirect(url_for('dashboard'))

    # Supprimer le fichier du dossier
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # Supprimer de la base de données
    db.session.delete(upload)
    db.session.commit()
    flash("Image supprimée avec succès.", 'success')
    return redirect(url_for('dashboard'))

@app.route('/feedback/<int:upload_id>', methods=['POST'])
def feedback(upload_id):
    if 'user_id' not in session:
        flash("Vous devez être connecté pour donner un feedback.", 'warning')
        return redirect(url_for('signin'))

    upload = Upload.query.get_or_404(upload_id)

    if upload.user_id != session['user_id']:
        flash("Action non autorisée.", 'danger')
        return redirect(url_for('dashboard'))

    try:
        rating = int(request.form.get('rating'))
        if rating not in (0, 1):
            raise ValueError
    except (TypeError, ValueError):
        flash("Note invalide.", 'warning')
        return redirect(url_for('dashboard'))

    upload.rating = rating

    # === CAS : Feedback négatif → Correction du plat
    if rating == 0:
        correction = request.form.get("correction", "").strip().lower().replace(" ", "_")
        if correction:
            upload.dish_name = correction

            # === Vérifie si le plat est déjà connu
            classes_path = os.path.join(app.root_path, 'classes_food5.txt')
            with open(classes_path, "r") as f:
                known_classes = [line.strip() for line in f.readlines()]

            # === Copie l'image dans le dossier du plat
            img_dir = os.path.join("dataset", "images", correction)
            os.makedirs(img_dir, exist_ok=True)

            src = os.path.join(app.config["UPLOAD_FOLDER"], upload.filename)
            dst = os.path.join(img_dir, upload.filename)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"[ERREUR] Copie image plat corrigé : {e}")

            # === Ajout à classes_food5.txt si nouveau
            if correction not in known_classes:
                with open(classes_path, "a") as f:
                    f.write(f"{correction}\n")

            # === Ajout dans plats.csv si nouveau
            plats_file = os.path.join(app.root_path, "plats.csv")
            correction_exists = False
            if os.path.exists(plats_file):
                with open(plats_file, newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0] == correction:
                            correction_exists = True
                            break
            if not correction_exists:
                with open(plats_file, "a", newline="") as fw:
                    writer = csv.writer(fw)
                    if os.path.getsize(plats_file) == 0:
                        writer.writerow(["name", "name_fr", "kcal", "protein_g", "carbs_g", "fat_g"])
                    writer.writerow([correction, correction.replace("_", " ").title(), 0, 0, 0, 0])

            # === Lancement du réentraînement toutes les 10 corrections
            json_path = os.path.join(app.root_path, "last_trained.json")
            if not os.path.exists(json_path):
                train_data = {"total_corrections": 0}
            else:
                with open(json_path, "r") as f:
                    try:
                        train_data = json.load(f)
                    except json.JSONDecodeError:
                        train_data = {"total_corrections": 0}

            # On incrémente le compteur
            train_data["total_corrections"] = train_data.get("total_corrections", 0) + 1
            corrections = train_data["total_corrections"]

            # On lance l'entraînement tous les 10 feedbacks négatifs
            if corrections % 10 == 0:
                try:
                    subprocess.Popen(["python", os.path.join(app.root_path, "train_from_scratch.py")])
                    flash("✔ Réentraînement automatique lancé après 10 corrections !", "info")
                except Exception as e:
                    flash("Erreur lors du réentraînement automatique.", "danger")
                    print(f"[ERREUR] train_from_scratch : {e}")

            # Enregistrer les données mises à jour
            with open(json_path, "w") as f:
                json.dump(train_data, f, indent=2)

    # === CAS : Feedback positif
    elif rating == 1:
        if upload.dish_name:
            dish_dir = os.path.join("dataset", "images", upload.dish_name.strip().lower().replace(" ", "_"))
            os.makedirs(dish_dir, exist_ok=True)

            src = os.path.join(app.config["UPLOAD_FOLDER"], upload.filename)
            dst = os.path.join(dish_dir, upload.filename)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"[ERREUR] Copie image feedback 👍 échouée : {e}")

    # === Enregistrement final
    db.session.commit()

    # === Backup dans feedback_data
    feedback_dir = os.path.join("feedback_data", str(rating))
    os.makedirs(feedback_dir, exist_ok=True)
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
    dst_path = os.path.join(feedback_dir, upload.filename)
    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"[ERREUR] Copie vers feedback_data échouée : {e}")

    # === Log CSV
    predicted_dish = upload.dish_name or "Inconnu"
    confidence = session.get("last_confidence", -1)
    with open(FEEDBACK_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([upload.filename, predicted_dish, confidence, rating])

    flash("Merci pour votre retour !", "success")
    return redirect(url_for('dashboard'))

#boutons barres supérieure
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        flash("Veuillez vous connecter pour accéder aux statistiques.", 'warning')
        return redirect(url_for('signin'))

    user_id = session.get('user_id')
    user_email = session.get('user_email')
    plats_csv_path = os.path.join(app.root_path, 'plats.csv')

    # Initialisation des totaux
    total_kcal = 0
    total_proteines = 0
    total_glucides = 0
    total_lipides = 0

    # Pour graphique 7 jours
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    kcal_per_day = {day: 0 for day in day_labels}

    # Calories de référence depuis UserProfile
    calories_needed = None
    calories_percentage = None
    profile = UserProfile.query.filter_by(user_email=user_email).first()

    if profile:
        try:
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + (5 if profile.sex == 'male' else -161)
            calories_needed = round(bmr * profile.activity)
        except Exception as e:
            print("Erreur calcul BMR :", e)

    if os.path.isfile(plats_csv_path):
        df = pd.read_csv(plats_csv_path)

        # Normalisation des noms
        if 'name' in df.columns:
            df['name_fr'] = df['name_fr'].str.lower().str.strip()
        else:
            flash("Le fichier plats.csv ne contient pas la colonne 'name'.", 'danger')
            return redirect(url_for('dashboard'))

        uploads = Upload.query.filter_by(user_id=user_id).all()

        for u in uploads:
            if not u.dish_name:
                continue
            dish_normalized = u.dish_name.lower().strip()
            match = df[df['name_fr'] == dish_normalized]
            if not match.empty:
                plat = match.iloc[0]
                total_kcal += float(plat.get('kcal', 0))
                total_proteines += float(plat.get('protein_g', 0))
                total_glucides += float(plat.get('carbs_g', 0))
                total_lipides += float(plat.get('fat_g', 0))

                day_str = u.timestamp.strftime('%a')  # Ex: "Mon"
                if day_str in kcal_per_day:
                    kcal_per_day[day_str] += float(plat.get('kcal', 0))

    # Calcul du pourcentage de consommation calorique
    if calories_needed:
        calories_percentage = round((total_kcal / calories_needed) * 100)

    return render_template('analytics.html',
        calories=int(round(total_kcal)),
        proteines=int(round(total_proteines)),
        glucides=int(round(total_glucides)),
        lipides=int(round(total_lipides)),
        calories_by_day=[int(k) for k in [kcal_per_day[day] for day in day_labels]],
        calories_needed=calories_needed,
        calories_percentage=calories_percentage
    )


@app.route('/training_status')
def training_status():
    import json
    status_file = "training_status.json"
    if not os.path.exists(status_file):
        return {"epoch": 0, "total": 1, "done": False}
    with open(status_file, "r") as f:
        return json.load(f)


# Déconnexion
@app.route('/logout')
def logout():
    session.clear()
    flash('Déconnexion réussie.', 'success')
    return redirect(url_for('signin'))

# ============ INITIALISATION ============

# Création automatique de la base de données
with app.app_context():
    db.create_all()

# Lancement de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
