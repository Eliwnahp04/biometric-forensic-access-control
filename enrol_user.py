import os
import pickle
import numpy as np
from deepface import DeepFace
from cryptography.fernet import Fernet

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Add users here. Point each to their raw photo.
# After enrolment is complete you can delete the raw images.

USERS_TO_ENROL = {
    "admin": {
        "raw_image": "images/admin.jpg",
    },
    "investigator": {
        "raw_image": "images/investigator.jpg",
    }
}

# All four models will be enrolled for every user.
# This lets you switch models freely in the GUI without re-enrolling.
MODELS = ["VGG-Face", "Facenet", "Facenet512", "ArcFace"]

# ── PATHS ─────────────────────────────────────────────────────────────────────

base_dir       = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(base_dir, "embeddings")
key_file       = os.path.join(embeddings_dir, "secret.key")

os.makedirs(embeddings_dir, exist_ok=True)

# ── ENCRYPTION KEY ────────────────────────────────────────────────────────────
# Generate a new key if one doesn't exist yet.
# If a key already exists it will be reused so existing .enc files stay valid.

if not os.path.exists(key_file):
    key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(key)
    print("New encryption key generated.\n")
else:
    with open(key_file, "rb") as f:
        key = f.read()
    print("Existing encryption key loaded.\n")

fernet = Fernet(key)

# ── ENROLMENT ─────────────────────────────────────────────────────────────────

for username, info in USERS_TO_ENROL.items():

    raw_image_path = os.path.join(base_dir, info["raw_image"])

    if not os.path.exists(raw_image_path):
        print(f"[SKIP] {username} — image not found at {raw_image_path}")
        print(f"       Make sure the image exists before running enrolment.\n")
        continue

    print(f"─── Enrolling: {username} ───────────────────────────────────────")

    all_embeddings = {}

    for model_name in MODELS:
        try:
            print(f"  [{model_name}] Extracting embedding...", end=" ")

            embedding_objs = DeepFace.represent(
                img_path=raw_image_path,
                model_name=model_name,
                enforce_detection=True
            )

            embedding = embedding_objs[0]["embedding"]
            all_embeddings[model_name] = embedding

            print(f"Done  ({len(embedding)} dimensions)")

        except Exception as e:
            print(f"FAILED — {str(e)}")
            print(f"         Skipping {model_name} for {username}.")

    if not all_embeddings:
        print(f"  No embeddings generated for {username}. Check the image.\n")
        continue

    # Serialise the full dict of embeddings (all models) and encrypt it
    data = pickle.dumps({
        "username":   username,
        "embeddings": all_embeddings   # { "VGG-Face": [...], "Facenet": [...], ... }
    })
    encrypted_data = fernet.encrypt(data)

    # Save as a single .enc file per user containing all model embeddings
    out_path = os.path.join(embeddings_dir, f"{username}.enc")
    with open(out_path, "wb") as f:
        f.write(encrypted_data)

    models_saved = list(all_embeddings.keys())
    print(f"\n  Saved: {out_path}")
    print(f"  Models stored: {', '.join(models_saved)}")
    print(f"  You can now delete: {raw_image_path}\n")

print("─────────────────────────────────────────────────────────────────────")
print("Enrolment complete.")
print("You can now switch between models freely in the GUI without re-enrolling.")