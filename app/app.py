from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, UnidentifiedImageError
from predict import model, transform, class_labels, DEVICE
import os
import secrets
import uuid
import requests
import shutil
import csv
import datetime
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'super_secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
migrate = Migrate(app, db)

EDAMAM_APP_ID = "0c070cdf"
EDAMAM_APP_KEY = "9749ef944be8a4c9d2ac6b659dcb2a09"

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    reset_token = db.Column(db.String(256), nullable=True)

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

def get_nutrition_from_food(food_query):
    parser_url = "https://api.edamam.com/api/food-database/v2/parser"
    nutrient_url = "https://api.edamam.com/api/food-database/v2/nutrients"
    params = {"app_id": EDAMAM_APP_ID, "app_key": EDAMAM_APP_KEY, "ingr": food_query}
    parser_response = requests.get(parser_url, params=params)
    if parser_response.status_code != 200:
        print("Erreur parser:", parser_response.text)
        return None
    data = parser_response.json()
    try:
        food_id = data['parsed'][0]['food']['foodId']
        label = data['parsed'][0]['food']['label']
    except (IndexError, KeyError):
        print("Aucun aliment trouvé.")
        return None
    payload = {
        "ingredients": [
            {
                "quantity": 1,
                "measureURI": "http://www.edamam.com/ontologies/edamam.owl#Measure_serving",
                "foodId": food_id
            }
        ]
    }
    nutrient_params = {"app_id": EDAMAM_APP_ID, "app_key": EDAMAM_APP_KEY}
    nutrient_response = requests.post(nutrient_url, params=nutrient_params, json=payload)
    if nutrient_response.status_code != 200:
        print("Erreur nutrients:", nutrient_response.text)
        return None
    nutrition_data = nutrient_response.json()
    nutrition_data["label"] = label
    return nutrition_data

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
    return render_template('dashboard.html',
                           uploads=uploads,
                           nutrition=session.get('last_nutrition'),
                           image_url=session.get('last_image'),
                           detected_dish=session.get('last_dish'),
                           corrected_id=session.pop('corrected_id', None))


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

        # ✅ Vérification de compatibilité
        try:
            with Image.open(filepath) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            os.remove(filepath)
            flash("Image non compatible. Veuillez choisir un fichier image valide (JPG, PNG).", "danger")
            return redirect(url_for('dashboard'))

        # ✅ Prédiction + confiance
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

        # ✅ API nutrition
        nutrition = get_nutrition_from_food(label_formatted)

        # ✅ Enregistrement DB
        new_upload = Upload(
            filename=filename,
            user_id=session['user_id'],
            dish_name=label_formatted
        )
        db.session.add(new_upload)
        db.session.commit()

        # ✅ Session
        session['last_nutrition'] = nutrition
        session['last_image'] = os.path.join('uploads', filename)
        session['last_dish'] = label_formatted
        session['last_confidence'] = round(confidence, 4)

        # ✅ Historique
        uploads = Upload.query.filter_by(user_id=session['user_id']).order_by(Upload.timestamp.desc()).all()

        # ✅ Pré-écriture du log feedback (sans rating pour l’instant)
        log_path = "feedback_csv/feedback_log.csv"
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "plat_prédit", "note", "correction", "date", "confiance", "re-entrainement"])

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                filename, label_formatted, "", "", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(confidence, 4), ""
            ])

        flash('Image bien reçue !', 'success')
        flash(f"Plat détecté : {label_formatted}", "info")

        session['last_image'] = os.path.join('uploads', filename)

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

    label = upload.dish_name
    nutrition = get_nutrition_from_food(label)

    session['last_nutrition'] = nutrition
    session['last_image'] = os.path.join('uploads', upload.filename)
    session['last_dish'] = label

    return redirect(url_for('dashboard'))


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

    # Vérifie que l'utilisateur est bien le propriétaire
    if upload.user_id != session['user_id']:
        flash("Action non autorisée.", 'danger')
        return redirect(url_for('dashboard'))

    try:
        rating = int(request.form.get('rating'))
        if rating < 1 or rating > 5:
            raise ValueError
    except (TypeError, ValueError):
        flash("Note invalide.", 'warning')
        return redirect(url_for('dashboard'))

    # Sauvegarde en base
    upload.rating = rating
    db.session.commit()

    # Créer le dossier feedback_data/<rating>/ si nécessaire
    feedback_dir = os.path.join("feedback_data", str(rating))
    os.makedirs(feedback_dir, exist_ok=True)

    # Copier l’image dans feedback_data/<rating>/
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
    dst_path = os.path.join(feedback_dir, upload.filename)
    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"[ERREUR] Copie vers feedback_data échouée : {e}")

    # Ajouter une ligne dans feedback_log.csv
    log_file = "feedback_log.csv"
    predicted_dish = upload.dish_name or "Inconnu"
    confidence = session.get("last_confidence", -1)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([upload.filename, predicted_dish, confidence, rating])

    flash("Merci pour votre retour !", "success")
    return redirect(url_for('dashboard'))

@app.route('/correct/<int:upload_id>', methods=['POST'])
def correct_label(upload_id):
    if 'user_id' not in session:
        flash("Vous devez être connecté.", 'warning')
        return redirect(url_for('signin'))

    upload = Upload.query.get_or_404(upload_id)

    if upload.user_id != session['user_id']:
        flash("Action non autorisée.", 'danger')
        return redirect(url_for('dashboard'))

    corrected = request.form.get('new_dish')
    if corrected:
        upload.dish_name = corrected.strip().lower()
        db.session.commit()
        flash("Nom du plat mis à jour ✅", "success")
        session['corrected_id'] = upload_id

    return redirect(url_for('dashboard'))


@app.route('/correction/<int:upload_id>', methods=['POST'])
def correction(upload_id):
    if 'user_id' not in session:
        flash("Accès non autorisé.", 'danger')
        return redirect(url_for('signin'))

    upload = Upload.query.get_or_404(upload_id)
    if upload.user_id != session['user_id']:
        flash("Action non autorisée.", 'danger')
        return redirect(url_for('dashboard'))

    new_dish = request.form.get("corrected_name", "").strip().lower().replace(" ", "_")
    if not new_dish:
        flash("Nom du plat invalide.", 'warning')
        return redirect(url_for('dashboard'))

    # Mise à jour du nom du plat
    upload.dish_name = new_dish.title()
    db.session.commit()
    flash("Nom du plat mis à jour.", 'success')

    # Copier l'image dans retraining_dataset/<new_dish>/
    save_corrected_image(upload.filename, new_dish)

    # Déclenche le réentraînement si assez d'exemples corrigés
    corrected_dir = os.path.join("retraining_dataset", new_dish)
    if os.path.exists(corrected_dir) and len(os.listdir(corrected_dir)) >= 10:
        import subprocess
        try:
            subprocess.run(["python", "retrain.py"], check=True)
            flash("✔ Réentraînement déclenché avec les corrections utilisateurs.", "info")
        except Exception as e:
            flash("❌ Échec du réentraînement automatique.", "danger")
            print(f"[ERREUR] Réentraînement : {e}")

    return redirect(url_for('dashboard'))


def save_corrected_image(filename, label):
    src = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dest_dir = os.path.join("retraining_dataset", label)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    try:
        shutil.copy(src, dest)
        print(f"[DEBUG] Image copiée dans : {dest}")
    except Exception as e:
        print(f"[ERREUR] Copie de l'image échouée : {e}")

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

if __name__ == '__main__':
    app.run(debug=True)
