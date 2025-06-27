import os
import shutil
from app import app, db, Upload

EXPORT_DIR = "feedback_data_errors"

with app.app_context():
    uploads = Upload.query.filter(Upload.rating <= 3).all()

    # Nettoyage
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)

    for upload in uploads:
        rating_folder = os.path.join(EXPORT_DIR, str(upload.rating))
        os.makedirs(rating_folder, exist_ok=True)

        src = os.path.join(app.config['UPLOAD_FOLDER'], upload.filename)
        if os.path.exists(src):
            dst = os.path.join(rating_folder, upload.filename)
            shutil.copyfile(src, dst)

    print("✅ Seules les images avec feedback ≤ 3 ont été exportées.")
