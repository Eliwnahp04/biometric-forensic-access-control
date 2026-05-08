import cv2
import sqlite3
import pickle
import numpy as np
from deepface import DeepFace
from cryptography.fernet import Fernet
from datetime import datetime
import os
import hashlib
import tkinter as tk
from tkinter import messagebox, Listbox, END, ttk

# -----------------------------
# Helper functions
# -----------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

def log_event(cursor, timestamp, username, role, event_type, outcome,
              attempt_number=None, similarity=None, file_accessed="NOT_ACCESSED",
              file_hash="NOT_GENERATED", details="", model_used=None, threshold_used=None):
    cursor.execute("""
        INSERT INTO access_log (
            timestamp, username, role, event_type, outcome,
            attempt_number, similarity, file_accessed, file_hash, details,
            model_used, threshold_used
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, username, role, event_type, outcome,
        attempt_number, similarity, file_accessed, file_hash, details,
        model_used, threshold_used
    ))

def cosine_distance(vec1, vec2):
    """Calculate cosine distance between two embedding vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cosine_similarity

# -----------------------------
# Eye Aspect Ratio (EAR) for liveness detection
# Based on Soukupova and Cech (2016)
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# A blink is detected when EAR drops below EAR_THRESHOLD
# for at least EAR_CONSEC_FRAMES consecutive frames
# -----------------------------

def eye_aspect_ratio(eye_points):
    """Calculate the Eye Aspect Ratio for a set of 6 eye landmark points."""
    A = np.linalg.norm(eye_points[1] - eye_points[5])  # vertical
    B = np.linalg.norm(eye_points[2] - eye_points[4])  # vertical
    C = np.linalg.norm(eye_points[0] - eye_points[3])  # horizontal
    return (A + B) / (2.0 * C)

# dlib 68-point landmark indices for left and right eye
LEFT_EYE_IDX   = list(range(42, 48))
RIGHT_EYE_IDX  = list(range(36, 42))

EAR_THRESHOLD     = 0.25   # EAR below this = eye closed
EAR_CONSEC_FRAMES = 2      # frames below threshold to confirm blink
LIVENESS_TIMEOUT  = 10     # seconds to wait for blink

def run_liveness_check(username, role):
    """
    Opens the webcam and monitors for an eye blink using the EAR method.
    Returns True if a blink is detected within LIVENESS_TIMEOUT seconds.
    Returns False if timed out — possible presentation attack.
    """
    import dlib
    import time

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(predictor_path):
        log_event(
            cursor, timestamp, username, role,
            event_type="LIVENESS_CHECK",
            outcome="ERROR",
            details="shape_predictor_68_face_landmarks.dat not found."
        )
        conn.commit()
        messagebox.showerror(
            "Liveness Error",
            "Facial landmark model not found.\n\n"
            "Please download:\nshape_predictor_68_face_landmarks.dat\n\n"
            "from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            "and place it in the same folder as this script."
        )
        return False

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        log_event(
            cursor, timestamp, username, role,
            event_type="LIVENESS_CHECK",
            outcome="FAILED",
            details="Webcam could not be opened for liveness check."
        )
        conn.commit()
        messagebox.showerror("Webcam Error", "Could not open webcam for liveness check.")
        return False

    blink_count    = 0
    consec_below   = 0
    blink_detected = False
    start_time     = time.time()

    print("Liveness check active — please blink naturally.")

    while True:
        elapsed = time.time() - start_time
        if elapsed > LIVENESS_TIMEOUT:
            break

        ret, frame = camera.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape     = predictor(gray, face)
            landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            left_ear  = eye_aspect_ratio(landmarks[LEFT_EYE_IDX])
            right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE_IDX])
            avg_ear   = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                consec_below += 1
            else:
                if consec_below >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                    if blink_count >= 1:
                        blink_detected = True
                consec_below = 0

        # Overlay feedback on webcam window
        remaining   = max(0, int(LIVENESS_TIMEOUT - elapsed))
        status_text = "BLINK DETECTED!" if blink_detected else f"Please blink... ({remaining}s)"
        colour      = (0, 255, 0) if blink_detected else (0, 165, 255)
        cv2.putText(frame, status_text,    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(frame, "Liveness Check", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Liveness Detection", frame)
        cv2.waitKey(1)

        if blink_detected:
            break

    camera.release()
    cv2.destroyAllWindows()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if blink_detected:
        log_event(
            cursor, timestamp, username, role,
            event_type="LIVENESS_CHECK",
            outcome="PASSED",
            details=f"Blink detected. EAR_THRESHOLD={EAR_THRESHOLD}, timeout={LIVENESS_TIMEOUT}s."
        )
        conn.commit()
        return True
    else:
        log_event(
            cursor, timestamp, username, role,
            event_type="LIVENESS_CHECK",
            outcome="FAILED",
            details=f"No blink detected within {LIVENESS_TIMEOUT}s. Possible presentation attack."
        )
        conn.commit()
        return False

# -----------------------------
# Paths and setup
# -----------------------------

base_dir        = os.path.dirname(os.path.abspath(__file__))
captured_image  = os.path.join(base_dir, "captured_user.jpg")
evidence_folder = os.path.join(base_dir, "forensic_evidence")
db_path         = os.path.join(base_dir, "mfa_access_logs_forensic.db")
embeddings_dir  = os.path.join(base_dir, "embeddings")
key_file        = os.path.join(embeddings_dir, "secret.key")

# -----------------------------
# Load encryption key
# -----------------------------

if not os.path.exists(key_file):
    messagebox.showerror(
        "Setup Error",
        "Encryption key not found.\nPlease run enrol_user.py first."
    )
    raise SystemExit

with open(key_file, "rb") as f:
    fernet = Fernet(f.read())

# -----------------------------
# Users dictionary
# -----------------------------

users = {
    "admin": {
        "password_hash": hash_password("Admin1234"),
        "embedding_file": os.path.join(embeddings_dir, "admin.enc"),
        "role": "admin"
    },
    "investigator": {
        "password_hash": hash_password("Secure5678"),
        "embedding_file": os.path.join(embeddings_dir, "investigator.enc"),
        "role": "investigator"
    }
}

# -----------------------------
# Available face recognition models
# -----------------------------

AVAILABLE_MODELS = ["VGG-Face", "Facenet", "Facenet512", "ArcFace"]

DEFAULT_THRESHOLDS = {
    "VGG-Face":   0.40,
    "Facenet":    0.30,
    "Facenet512": 0.30,
    "ArcFace":    0.68,
}

# -----------------------------
# Database setup
# -----------------------------

conn   = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS access_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT,
    username         TEXT,
    role             TEXT,
    event_type       TEXT,
    outcome          TEXT,
    attempt_number   INTEGER,
    similarity       REAL,
    file_accessed    TEXT,
    file_hash        TEXT,
    details          TEXT,
    model_used       TEXT,
    threshold_used   REAL
)
""")

existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(access_log)")]
if "model_used" not in existing_cols:
    cursor.execute("ALTER TABLE access_log ADD COLUMN model_used TEXT")
if "threshold_used" not in existing_cols:
    cursor.execute("ALTER TABLE access_log ADD COLUMN threshold_used REAL")

conn.commit()

# -----------------------------
# App state
# -----------------------------

attempts        = 0
max_attempts    = 3
current_user    = None
current_role    = "UNKNOWN"
allowed_files   = []
last_similarity = None

# -----------------------------
# Load encrypted embedding
# -----------------------------

def load_embedding(username, model_name):
    """Decrypt and return the stored face embedding for a user and model."""
    enc_path = users[username]["embedding_file"]
    if not os.path.exists(enc_path):
        return None
    with open(enc_path, "rb") as f:
        encrypted_data = f.read()
    data = pickle.loads(fernet.decrypt(encrypted_data))
    embeddings = data.get("embeddings", {})
    if model_name not in embeddings:
        messagebox.showerror(
            "Model Not Enrolled",
            f"No embedding found for model '{model_name}'.\n"
            f"Please re-run enrol_user.py to enrol all models."
        )
        return None
    return embeddings[model_name]

# -----------------------------
# Face verification with optional liveness check
# Authentication pipeline:
#   1. Password check (login function)
#   2. Liveness check — EAR blink detection (if enabled)
#   3. Face verification — DeepFace embedding comparison
# -----------------------------

def run_face_auth(username, role):
    global last_similarity

    selected_model     = model_var.get()
    selected_threshold = float(threshold_var.get())
    auth_mode          = mode_var.get()
    timestamp          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Password-only mode — skip all biometric checks
    if auth_mode == "Password Only":
        log_event(
            cursor, timestamp, username, role,
            event_type="FACE_VERIFICATION",
            outcome="SKIPPED",
            details="Password-only mode selected. Face verification skipped.",
            model_used="N/A",
            threshold_used=None
        )
        conn.commit()
        return True

    # --- Step 1: Liveness check (if enabled) ---
    if liveness_var.get():
        messagebox.showinfo(
            "Liveness Check",
            "Liveness check starting.\n\n"
            "Please look at the webcam and blink naturally.\n"
            "You have 10 seconds."
        )
        liveness_passed = run_liveness_check(username, role)
        if not liveness_passed:
            messagebox.showerror(
                "Liveness Failed",
                "Liveness check failed.\n"
                "No blink detected within the time limit.\n"
                "Possible spoofing attempt. Access denied."
            )
            return False
        messagebox.showinfo(
            "Liveness Passed",
            "Liveness check passed.\nProceeding to face verification."
        )

    # --- Step 2: Webcam capture ---
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        log_event(
            cursor, timestamp, username, role,
            event_type="WEBCAM",
            outcome="FAILED",
            details="Webcam could not be opened.",
            model_used=selected_model,
            threshold_used=selected_threshold
        )
        conn.commit()
        messagebox.showerror("Webcam Error", "Could not open webcam.")
        return False

    print("Press SPACE to capture your face")

    while True:
        ret, frame = camera.read()
        if not ret:
            camera.release()
            cv2.destroyAllWindows()
            messagebox.showerror("Webcam Error", "Failed to read from webcam.")
            return False

        cv2.imshow("Face Verification — Press SPACE to capture", frame)
        key = cv2.waitKey(1)

        if key == 32:
            cv2.imwrite(captured_image, frame)
            break

    camera.release()
    cv2.destroyAllWindows()

    # --- Step 3: Embedding comparison ---
    try:
        stored_embedding = load_embedding(username, selected_model)
        if stored_embedding is None:
            return False

        live_objs = DeepFace.represent(
            img_path=captured_image,
            model_name=selected_model,
            enforce_detection=True
        )
        live_embedding  = live_objs[0]["embedding"]
        distance        = cosine_distance(stored_embedding, live_embedding)
        last_similarity = round(distance, 4)

        similarity_label.config(
            text=f"Similarity Score: {last_similarity}  |  Threshold: {selected_threshold}  |  Model: {selected_model}"
        )

        verified       = distance <= selected_threshold
        liveness_note  = f" Liveness={'ON' if liveness_var.get() else 'OFF'}."

        if verified:
            log_event(
                cursor, timestamp, username, role,
                event_type="FACE_VERIFICATION",
                outcome="PASSED",
                similarity=last_similarity,
                details=(
                    f"Face verified. Distance={last_similarity}, "
                    f"Threshold={selected_threshold}, Model={selected_model}."
                    f"{liveness_note}"
                ),
                model_used=selected_model,
                threshold_used=selected_threshold
            )
            conn.commit()
            return True
        else:
            log_event(
                cursor, timestamp, username, role,
                event_type="FACE_VERIFICATION",
                outcome="FAILED",
                similarity=last_similarity,
                details=(
                    f"Face not verified. Distance={last_similarity}, "
                    f"Threshold={selected_threshold}, Model={selected_model}."
                    f"{liveness_note}"
                ),
                model_used=selected_model,
                threshold_used=selected_threshold
            )
            conn.commit()
            return False

    except Exception as e:
        log_event(
            cursor, timestamp, username, role,
            event_type="FACE_VERIFICATION",
            outcome="ERROR",
            details=f"DeepFace error: {str(e)}",
            model_used=selected_model,
            threshold_used=selected_threshold
        )
        conn.commit()
        messagebox.showerror("Face Verification Error", str(e))
        return False

    finally:
        if os.path.exists(captured_image):
            os.remove(captured_image)

# -----------------------------
# Evidence window
# -----------------------------

def open_selected_file():
    selection = file_listbox.curselection()
    if not selection:
        messagebox.showwarning("No Selection", "Please select a file.")
        return

    selected_name = allowed_files[selection[0]]
    selected_file = os.path.join(evidence_folder, selected_name)
    timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_hash     = generate_file_hash(selected_file)

    log_event(
        cursor, timestamp, current_user, current_role,
        event_type="FILE_OPENED",
        outcome="GRANTED",
        similarity=last_similarity,
        file_accessed=selected_name,
        file_hash=file_hash,
        details=f"Evidence file opened by role '{current_role}'."
    )
    conn.commit()

    os.startfile(selected_file)
    messagebox.showinfo("File Opened", f"{selected_name}\n\nSHA-256:\n{file_hash}")

def show_evidence_window():
    global file_listbox, allowed_files

    evidence_window = tk.Toplevel(root)
    evidence_window.title("Forensic Evidence Access")
    evidence_window.geometry("520x420")

    auth_mode = mode_var.get()

    tk.Label(
        evidence_window,
        text=f"Authenticated as: {current_user}  |  Role: {current_role}  |  Mode: {auth_mode}",
        font=("Arial", 10, "bold")
    ).pack(pady=10)

    if auth_mode == "MFA" and last_similarity is not None:
        liveness_status = "ON" if liveness_var.get() else "OFF"
        tk.Label(
            evidence_window,
            text=(
                f"Face distance: {last_similarity}  |  "
                f"Model: {model_var.get()}  |  "
                f"Threshold: {threshold_var.get()}  |  "
                f"Liveness: {liveness_status}"
            ),
            font=("Arial", 9),
            fg="darkgreen"
        ).pack()

    files = os.listdir(evidence_folder) if os.path.exists(evidence_folder) else []

    if current_role == "admin":
        allowed_files = files
    elif current_role == "investigator":
        allowed_files = [f for f in files if f != "hash_log.txt"]
    else:
        allowed_files = []

    if not allowed_files:
        tk.Label(evidence_window, text="No files available for this role.").pack(pady=20)
        return

    tk.Label(evidence_window, text="Available forensic evidence files:").pack(pady=5)

    file_listbox = Listbox(evidence_window, width=65, height=10)
    file_listbox.pack(pady=10)

    for f in allowed_files:
        file_listbox.insert(END, f)

    tk.Button(
        evidence_window,
        text="Open Selected File",
        command=open_selected_file,
        width=20
    ).pack(pady=10)

# -----------------------------
# Login logic
# -----------------------------

def login():
    global attempts, current_user, current_role

    username  = username_entry.get().strip()
    password  = password_entry.get().strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_event(
        cursor, timestamp, username, "UNKNOWN",
        event_type="LOGIN_ATTEMPT",
        outcome="STARTED",
        attempt_number=attempts + 1,
        details=(
            f"User submitted credentials. Mode={mode_var.get()}, "
            f"Model={model_var.get()}, Threshold={threshold_var.get()}, "
            f"Liveness={'ON' if liveness_var.get() else 'OFF'}."
        ),
        model_used=model_var.get() if mode_var.get() == "MFA" else "N/A",
        threshold_used=float(threshold_var.get()) if mode_var.get() == "MFA" else None
    )
    conn.commit()

    if username not in users:
        attempts += 1
        log_event(
            cursor, timestamp, username, "UNKNOWN",
            event_type="USER_NOT_FOUND",
            outcome="FAILED",
            attempt_number=attempts,
            details="Username does not exist in the system."
        )
        conn.commit()
        messagebox.showerror("Login Failed", f"User not found.\nAttempts left: {max_attempts - attempts}")

    else:
        role = users[username]["role"]

        if hash_password(password) == users[username]["password_hash"]:
            log_event(
                cursor, timestamp, username, role,
                event_type="PASSWORD_CHECK",
                outcome="PASSED",
                attempt_number=attempts + 1,
                details="Password authentication successful."
            )
            conn.commit()

            current_user = username
            current_role = role

            mode = mode_var.get()
            if mode == "MFA":
                if liveness_var.get():
                    messagebox.showinfo(
                        "Password Check",
                        "Password correct.\nLiveness check will start, followed by face verification."
                    )
                else:
                    messagebox.showinfo(
                        "Password Check",
                        "Password correct.\nWebcam face verification will start now."
                    )
            else:
                messagebox.showinfo("Password Check", "Password correct.\nPassword-only mode active.")

            if run_face_auth(username, role):
                messagebox.showinfo("Access Granted", "Authentication successful. Access granted.")
                show_evidence_window()
            else:
                messagebox.showerror("Access Denied", "Authentication failed. Access denied.")

        else:
            attempts += 1
            log_event(
                cursor, timestamp, username, role,
                event_type="PASSWORD_CHECK",
                outcome="FAILED",
                attempt_number=attempts,
                details="Incorrect password entered."
            )
            conn.commit()
            messagebox.showerror("Login Failed", f"Incorrect password.\nAttempts left: {max_attempts - attempts}")

    if attempts >= max_attempts:
        log_event(
            cursor, timestamp, username, current_role,
            event_type="ACCOUNT_LOCKOUT",
            outcome="LOCKED",
            attempt_number=attempts,
            details="Maximum login attempts exceeded."
        )
        conn.commit()
        messagebox.showerror("Locked Out", "Too many failed attempts. Access locked.")
        login_button.config(state="disabled")

# -----------------------------
# Update threshold when model changes
# -----------------------------

def on_model_change(*args):
    model = model_var.get()
    threshold_var.set(str(DEFAULT_THRESHOLDS.get(model, 0.40)))

# -----------------------------
# GUI
# -----------------------------

root = tk.Tk()
root.title("Forensic Evidence Access System")
root.geometry("480x460")
root.resizable(False, False)

tk.Label(
    root,
    text="Forensic Evidence Access System",
    font=("Arial", 14, "bold")
).pack(pady=12)

# Username
tk.Label(root, text="Username").pack()
username_entry = tk.Entry(root, width=30)
username_entry.pack(pady=4)

# Password
tk.Label(root, text="Password").pack()
password_entry = tk.Entry(root, show="*", width=30)
password_entry.pack(pady=4)

# Settings frame
settings_frame = tk.Frame(root)
settings_frame.pack(pady=8)

# Auth mode
tk.Label(settings_frame, text="Auth Mode:").grid(row=0, column=0, padx=6, sticky="e")
mode_var = tk.StringVar(value="MFA")
mode_menu = ttk.Combobox(
    settings_frame,
    textvariable=mode_var,
    values=["MFA", "Password Only"],
    state="readonly",
    width=14
)
mode_menu.grid(row=0, column=1, padx=6)

# Model selector
tk.Label(settings_frame, text="Model:").grid(row=0, column=2, padx=6, sticky="e")
model_var = tk.StringVar(value="VGG-Face")
model_var.trace("w", on_model_change)
model_menu = ttk.Combobox(
    settings_frame,
    textvariable=model_var,
    values=AVAILABLE_MODELS,
    state="readonly",
    width=12
)
model_menu.grid(row=0, column=3, padx=6)

# Threshold
tk.Label(settings_frame, text="Threshold:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
threshold_var = tk.StringVar(value="0.40")
threshold_entry = tk.Entry(settings_frame, textvariable=threshold_var, width=8)
threshold_entry.grid(row=1, column=1, padx=6, sticky="w")
tk.Label(
    settings_frame,
    text="(lower = stricter)",
    font=("Arial", 8),
    fg="grey"
).grid(row=1, column=2, columnspan=2, sticky="w")

# Liveness detection toggle checkbox
liveness_var = tk.BooleanVar(value=True)
liveness_check = tk.Checkbutton(
    settings_frame,
    text="Enable Liveness Detection (blink check)",
    variable=liveness_var,
    font=("Arial", 9)
)
liveness_check.grid(row=2, column=0, columnspan=4, pady=6)

# Login button
login_button = tk.Button(root, text="Login", command=login, width=15)
login_button.pack(pady=10)

# Similarity score display
similarity_label = tk.Label(
    root,
    text="Similarity Score: —  |  Threshold: —  |  Model: —",
    font=("Arial", 9),
    fg="darkblue"
)
similarity_label.pack(pady=4)

root.mainloop()

conn.close()