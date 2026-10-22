import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print("-" * 50)

def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
        return True
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__} - {e}")
        return False

# 1. External packages
check("cv2", lambda: __import__("cv2"))
check("numpy", lambda: __import__("numpy"))
check("PIL", lambda: __import__("PIL"))
check("sounddevice", lambda: __import__("sounddevice"))
check("soundfile", lambda: __import__("soundfile"))
check("torch", lambda: __import__("torch"))
check("ultralytics", lambda: __import__("ultralytics"))
check("chromadb", lambda: __import__("chromadb"))
check("groq", lambda: __import__("groq"))

# 2. Local modules
check("config", lambda: __import__("config"))
check("robot_ui", lambda: __import__("robot_ui"))
check("qr_login", lambda: __import__("qr_login"))
check("classifier", lambda: __import__("classifier"))
check("rag.retriever", lambda: __import__("rag.retriever"))
check("rag.query_rewrite", lambda: __import__("rag.query_rewrite"))
check("rag.response_generator", lambda: __import__("rag.response_generator"))
check("audio.recorder", lambda: __import__("audio.recorder"))
check("audio.tts", lambda: __import__("audio.tts"))

print("-" * 50)
# 3. Model weights and data
from pathlib import Path
from config import CLASSIFIER_MODEL_PATH, CHROMA_PERSIST_DIR
print(f"CLASSIFIER_MODEL_PATH: {CLASSIFIER_MODEL_PATH} -> Exists: {CLASSIFIER_MODEL_PATH.exists()}")
print(f"CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR} -> Exists: {Path(CHROMA_PERSIST_DIR).exists()}")
from config import GROQ_API_KEY
print(f"GROQ_API_KEY available: {bool(GROQ_API_KEY or os.environ.get('GROQ_API_KEY'))}")
