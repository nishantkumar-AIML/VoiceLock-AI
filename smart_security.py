import os
import torch
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from speechbrain.inference.speaker import EncoderClassifier
from datetime import datetime, timedelta
import glob

# --- SETTINGS ---
os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
DB_FOLDER = "voice_db"     # Yahan users ka data save hoga
EXPIRY_DAYS = 7            # 7 din baad delete
THRESHOLD = 0.35           # Thoda strict kiya taaki duplicate users kam bane
RECORD_SECONDS = 5
SAMPLE_RATE = 16000

# Folder nahi hai to bana lo
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

print("\n🚀 Starting Smart Security System...")

# --- 1. MODEL LOAD ---
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    print("✅ Brain Loaded Successfully!\n")
except Exception as e:
    print(f"❌ Model Error: {e}")
    exit()

# --- 2. MEMORY MANAGEMENT FUNCTIONS ---

def cleanup_old_users():
    """Check karega ki kaun 7 din se gayab hai aur delete karega"""
    print("🧹 Cleaning up old data...")
    count = 0
    now = datetime.now()
    
    # DB folder ki saari files check karo
    for filepath in glob.glob(os.path.join(DB_FOLDER, "*.pt")):
        try:
            data = torch.load(filepath)
            last_seen = data['last_seen']
            
            # Agar 7 din se purana hai
            if (now - last_seen).days >= EXPIRY_DAYS:
                os.remove(filepath)
                print(f"   🗑️ Deleted expired user: {os.path.basename(filepath)}")
                count += 1
        except:
            pass # Agar file corrupt hai to chod do
            
    if count == 0:
        print("   ✅ No old users to delete.")
    else:
        print(f"   ✅ Cleaned {count} old profiles.")

def load_known_speakers():
    """Start hote waqt saved users ko memory mein layega"""
    speakers = []
    for filepath in glob.glob(os.path.join(DB_FOLDER, "*.pt")):
        try:
            data = torch.load(filepath)
            speakers.append({
                'id': data['id'],
                'emb': data['emb'],
                'path': filepath
            })
        except:
            print(f"⚠️ Corrupt file found: {filepath}")
    return speakers

def save_user_profile(user_id, embedding):
    """User ko Hard Disk par save/update karega"""
    filepath = os.path.join(DB_FOLDER, f"{user_id}.pt")
    data = {
        'id': user_id,
        'emb': embedding,
        'last_seen': datetime.now() # Abhi ka time note kar lo
    }
    torch.save(data, filepath)
    return filepath

# --- 3. PROCESSING LOGIC ---

# Step A: Purana kachra saaf karo
cleanup_old_users()

# Step B: Jo bache hain unhe load karo
known_speakers = load_known_speakers()
print(f"📂 Loaded {len(known_speakers)} users from database.")

def get_embedding_from_file(filename):
    try:
        audio, _ = librosa.load(filename, sr=16000)
        # Silence check (Agar audio khali hai)
        if np.max(np.abs(audio)) < 0.01:
            return None
        tensor = torch.tensor(audio).float().unsqueeze(0)
        return classifier.encode_batch(tensor)
    except:
        return None

def identify_and_update(new_emb):
    global known_speakers
    
    best_score = -1
    best_user = None
    best_index = -1

    # 1. Existing users se compare karo
    for i, speaker in enumerate(known_speakers):
        score = torch.nn.CosineSimilarity(dim=2)(new_emb, speaker['emb']).item()
        if score > best_score:
            best_score = score
            best_user = speaker
            best_index = i

    # 2. Decision Logic
    if best_score > THRESHOLD:
        # --- KNOWN USER (Update Time) ---
        user_id = best_user['id']
        
        # User ka 'Last Seen' time update karke wapas save karo
        # (Taaki wo agle 7 din tak delete na ho)
        save_user_profile(user_id, best_user['emb'])
        
        return user_id, best_score, False # False = Old User
    
    else:
        # --- NEW USER (Create File) ---
        # Naya ID banao (jitne files hain + 1)
        new_id = f"User_{len(known_speakers) + 1}"
        
        # Hard disk pe save karo
        filepath = save_user_profile(new_id, new_emb)
        
        # RAM mein bhi add karo
        known_speakers.append({'id': new_id, 'emb': new_emb, 'path': filepath})
        
        return new_id, best_score, True # True = New User

# --- 4. MAIN LOOP ---
try:
    while True:
        print(f"\n🔴 Listening ({RECORD_SECONDS}s)...")
        
        # Record
        recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        
        temp_file = "temp_live.wav"
        write(temp_file, SAMPLE_RATE, (recording * 32767).astype(np.int16))
        
        print("⚡ Processing...")
        current_emb = get_embedding_from_file(temp_file)
        
        if current_emb is not None:
            user_id, score, is_new = identify_and_update(current_emb)
            
            if is_new:
                print(f"✨ NEW VISITOR DETECTED -> Created Profile: [{user_id}]")
            else:
                print(f"✅ WELCOME BACK -> Identified: [{user_id}] (Score: {score:.2f})")
                print(f"   (Profile updated: Expiry extended by 7 days)")
        else:
            print("⚠️ Silence or Noise (Ignored)")

except KeyboardInterrupt:
    print("\n🛑 System Stopped.")
    if os.path.exists("temp_live.wav"):
        os.remove("temp_live.wav")