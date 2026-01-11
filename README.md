# 🎙️ VoiceLock AI - Smart Speaker Recognition System

VoiceLock is an advanced AI-based security system designed to authenticate users based on their unique voice fingerprints. Unlike traditional systems, it uses **Deep Learning (ECAPA-TDNN)** to identify speakers in real-time and manages user data dynamically.

## 🚀 Key Features

- **🧠 Deep Learning Core:** Powered by `SpeechBrain` and `PyTorch` for high-accuracy speaker verification.
- **🔄 Dynamic User Enrollment:** Automatically registers new users (User 1, User 2...) without manual setup.
- **🛡️ Smart Privacy:** Does **not** store raw audio files. Only mathematical embeddings (`.pt` tensors) are saved to protect user privacy.
- **🧹 Auto-Cleanup Logic:** Automatically deletes user profiles if they are inactive for **7 days** to save storage.
- **🌐 Flask API Integration:** Ready to connect with Android/Web apps via a lightweight REST API.

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **AI/ML:** PyTorch, SpeechBrain, Torchaudio
- **Audio Processing:** Librosa, NumPy
- **Backend API:** Flask
- **Hardware:** Optimized for CPU (Mac M4/Intel) & GPU

## ⚙️ How It Works

1.  **Input:** The system captures audio via Microphone or API upload.
2.  **Processing:** It converts audio into a waveform and extracts a **192-dimensional vector embedding**.
3.  **Matching:** It compares the live embedding with the database using **Cosine Similarity**.
    - If Score > 0.35 → **Access Granted ✅** (User Identified)
    - If Score < 0.35 → **New User Registered 🆕**
4.  **Maintenance:** On every launch, it scans and removes expired user profiles.

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/VoiceLock-AI.git](https://github.com/YOUR_USERNAME/VoiceLock-AI.git)
    cd VoiceLock-AI
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Server:
    ```bash
    python server.py
    ```

## 🔮 Future Scope
- Integration with Android App (Vani).
- Converting the model to TFLite for offline mobile usage.
- Adding "Wake Word" detection (e.g., "Hey Jarvis").

---
*Developed by Nishant Kumar | B.E. CSE (AI/ML)*