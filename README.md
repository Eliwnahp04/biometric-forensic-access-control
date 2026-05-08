# biometric-forensic-access-control
Thesis Project - Privacy-aware biometric authentication system for secure access to digital forensic evidence

# Privacy-Aware Biometric Authentication for Secure Access to Digital Forensic Evidence

**Author:** Elina Honarpisheh, B00157079  
**Institution:** Technological University Dublin  
**Programme:** B.Sc. Digital Forensics and Cyber Security  
**Supervisor:** Bahareh Pahlevandzadeh  
**Submission:** May 2026  

---

## Overview

This project builds and evaluates a biometric authentication system for controlling access to digital forensic evidence. The system uses a multi-factor pipeline combining password verification and facial recognition, with role-based access control and a forensic audit log designed to meet the chain-of-custody requirements of ISO/IEC 27037.

The privacy side of the design matters as much as the security side. Raw face photographs are used only during enrolment. After enrolment, they are deleted and replaced with AES-128 Fernet encrypted face embeddings. No raw biometric image stays on disk during normal operation.

---

## What the system does

- Password verification with SHA-256 hashing and a 3-attempt lockout
- EAR-based blink detection for liveness verification, blocking photo spoofing attacks
- Facial recognition via DeepFace, supporting VGG-Face, Facenet, Facenet512, and ArcFace
- Configurable acceptance threshold per model
- Role-based access control with administrator and investigator roles
- SHA-256 file integrity hashing recorded at the point of evidence access
- Full forensic audit log stored in a SQLite database
- AES-128 Fernet encrypted face embeddings, no raw photos stored on disk

---

## Project structure

```
biometric-forensic-access-control/
├── mfa_access_control_gui.py       # Main application
├── enrol_user.py                   # Enrolment script
├── forensic_evidence/              # Evidence files directory
├── embeddings/                     # Generated locally, not included
│   └── README.md
└── README.md
```

---

## Requirements

- Windows 10 or Windows 11 (64-bit)
- Python 3.10 or higher (tested on Python 3.12)
- Webcam
- 4 GB RAM minimum, 8 GB recommended
- Around 3 GB disk space for all dependencies

---

## Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/Eliwnahp04/biometric-forensic-access-control
cd biometric-forensic-access-control
```

**Step 2: Create and activate a virtual environment**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
pip install deepface dlib-bin cryptography opencv-python numpy
```

**Step 4: Download the facial landmark model**

Download `shape_predictor_68_face_landmarks.dat.bz2` from:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract it using 7-Zip or WinRAR and place `shape_predictor_68_face_landmarks.dat` in the project root directory.

**Step 5: Run enrolment**

```bash
python enrol_user.py
```

This generates the encrypted embedding files and the encryption key. Reference face photographs are deleted after enrolment finishes.

**Step 6: Run the application**

```bash
python mfa_access_control_gui.py
```

---

## Default credentials

| Username | Password | Role |
|----------|----------|------|
| admin | Admin1234 | Admin |
| investigator | Secure5678 | Investigator |

---

## Note on embeddings

The `embeddings/` folder is not included in this repository. Encrypted face embeddings and the encryption key are generated locally by running `enrol_user.py`. This is intentional: no biometric data is stored in the repository.

---

## Experiments

Six experiments were conducted to evaluate the system:

| Experiment | Focus | Key finding |
|------------|-------|-------------|
| 1 | Threshold sensitivity | Optimal thresholds: VGG-Face 0.30, Facenet 0.15, Facenet512 0.30, ArcFace 0.50 |
| 2 | Model comparison | VGG-Face and ArcFace both 100% reliable, ArcFace fastest at 1.12s average |
| 3 | Environmental variation | Lighting had minimal impact, head angle and glasses caused failures |
| 4 | Security attacks | Photo spoofing passed 4 out of 5 attempts without liveness detection |
| 5 | MFA vs password-only | MFA overhead 1.91 seconds, credential theft blocked at face verification stage |
| 6 | Liveness detection | Spoofing success rate dropped from 80% to 0% with liveness enabled |

---

## Troubleshooting

**DeepFace is slow on the first run.** The model loads into memory on the first authentication attempt. This takes 3 to 7 seconds. Subsequent attempts run in 1 to 2 seconds.

**Face not detected error.** Make sure the webcam is well lit and your face is centred in the frame. Head angles greater than about 30 degrees from frontal will cause detection failures, as confirmed in Experiment 3.

**Database is locked.** Close DB Browser for SQLite before running the application. SQLite only allows one write connection at a time.

**Liveness check times out immediately.** Check that `shape_predictor_68_face_landmarks.dat` is in the same directory as `mfa_access_control_gui.py`. The liveness check will fail immediately if the file is missing.

---

## License

This project was developed for academic purposes as part of a final year thesis at Technological University Dublin.
