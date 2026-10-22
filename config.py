"""
config.py — Cấu hình ThinkMed (HỢP NHẤT: RAG/LLM/Audio + Classifier).

Gộp từ hai nguồn:
  - Cấu hình RAG/Groq/audio (bản gốc)
  - Cấu hình tầng phân loại ảnh (classifier mới)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Tự động đọc file .env cục bộ nếu có
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════
DOCX_PATH          = BASE_DIR / "data" / "skin_old_new_docu.docx"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME    = "skin_diseases"
MODELS_DIR         = BASE_DIR / "models"

# ══════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL       = "intfloat/multilingual-e5-small"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "best-yolo11s.pt"

# ══════════════════════════════════════════════════════════════════════
#  LLM BACKEND
# ══════════════════════════════════════════════════════════════════════
# "groq" | "ollama" | "anthropic" | "none"
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001"

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# ══════════════════════════════════════════════════════════════════════
#  TTS BACKEND & API
# ══════════════════════════════════════════════════════════════════════
# "piper" (100% Offline trên RAM, Giọng Ban Mai siêu tốc ~0.12s)
# "edge" (Cloud API Microsoft Hoài My Neural tự nhiên)
# "elevenlabs" (Cloud API ElevenLabs Multilingual v2)
TTS_BACKEND = os.getenv("TTS_BACKEND", "piper").lower()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_432f504677b0d88518feebe9550810fa8fa0615e55b882c2")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Bella
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# ══════════════════════════════════════════════════════════════════════
#  RAG SETTINGS
# ══════════════════════════════════════════════════════════════════════
TOP_K_CHUNKS  = 5
MIN_CHUNK_LEN = 150

# ══════════════════════════════════════════════════════════════════════
#  CLASSIFIER — LỚP (best-yolo11s.pt có 5 lớp)
# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "acne",
    "atopic_dermatitis",
    "psoriasis",
    "tinea",
    "wart",
]
OTHER_CLASS = "other"

# class_name (model) -> disease_key (RAG). `other` KHÔNG có: nó dẫn tới "đi khám".
# CHÚ Ý: disease_key phải khớp metadata trong ChromaDB (build_rag_full.py).
CLASS_TO_DISEASE = {
    "acne":              "acne",
    "atopic_dermatitis": "atopic_dermatitis",
    "psoriasis":         "psoriasis",
    "tinea":             "tinea",
    "wart":              "wart",
}

# disease_key -> tên tiếng Việt. Gồm cả bệnh chỉ-hội-thoại (leprosy, KP, impetigo)
# mà model KHÔNG phân loại từ ảnh, nhưng RAG có tài liệu.
DISEASE_VI_NAMES = {
    "acne":              "Mụn trứng cá",
    "atopic_dermatitis": "Viêm da cơ địa",
    "psoriasis":         "Vẩy nến",
    "tinea":             "Nấm da",
    "wart":              "Mụn cóc do virus",
    "leprosy":           "Bệnh phong",
    "keratosis_pilaris": "Dày sừng nang lông",
    "impetigo":          "Chốc lở",
    "healthy":           "Da khỏe mạnh",
}

# ══════════════════════════════════════════════════════════════════════
#  PHÂN LUỒNG PHÒNG KHÁM (Gợi ý 4 lớp bệnh vào 4 phòng khám chuyên khoa)
# ══════════════════════════════════════════════════════════════════════
DISEASE_CLINIC_ROOMS = {
    "healthy": {
        "room_num": "Tư vấn",
        "room_name": "Chăm sóc & Dưỡng da",
        "location": "Khu Tiếp đón - Tầng 1",
        "voice_hint": "Da của bạn đang rất khỏe mạnh và không có dấu hiệu bệnh lý.",
    },
    "acne": {
        "room_num": "Phòng 101",
        "room_name": "Khám Da liễu & Thẩm mỹ da",
        "location": "Tầng 1 - Khu A",
        "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 101 (Khám Da liễu & Thẩm mỹ da) tại Tầng 1 để được bác sĩ chuyên khoa thăm khám.",
    },
    "atopic_dermatitis": {
        "room_num": "Phòng 102",
        "room_name": "Khám Dị ứng & Miễn dịch da",
        "location": "Tầng 1 - Khu A",
        "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 102 (Khám Dị ứng & Miễn dịch da) tại Tầng 1 để kiểm tra và nhận phác đồ phù hợp.",
    },
    "psoriasis": {
        "room_num": "Phòng 103",
        "room_name": "Khám Da liễu Mạn tính & Tự miễn",
        "location": "Tầng 1 - Khu B",
        "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 103 (Khám Da liễu Mạn tính) tại Tầng 1 để được bác sĩ theo dõi chuyên sâu.",
    },
    "tinea": {
        "room_num": "Phòng 104",
        "room_name": "Xét nghiệm Vi nấm & Ký sinh trùng",
        "location": "Tầng 1 - Khu B",
        "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 104 (Phòng Xét nghiệm Vi nấm) để soi tươi tìm nấm và điều trị dứt điểm.",
    },
    "wart": {
        "room_num": "Phòng 104",  # chung cụm phòng thủ thuật/nhiễm
        "room_name": "Khám & Thủ thuật Da liễu",
        "location": "Tầng 1 - Khu B",
        "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 104 (Khám & Thủ thuật Da liễu) để được kiểm tra và xử lý an toàn.",
    },
}

def get_clinic_room(disease_key: str) -> dict:
    """Trả về thông tin phòng khám gợi ý cho bệnh nhân theo từng loại bệnh."""
    if not disease_key:
        return {
            "room_num": "Phòng 100",
            "room_name": "Khám Da liễu Tổng quát",
            "location": "Tầng 1 - Khu Tiếp đón",
            "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 100 (Khám Da liễu Tổng quát) để được bác sĩ sàng lọc ban đầu.",
        }
    key = disease_key.lower().strip()
    return DISEASE_CLINIC_ROOMS.get(key, {
        "room_num": "Phòng 100",
        "room_name": "Khám Da liễu Tổng quát",
        "location": "Tầng 1 - Khu Tiếp đón",
        "voice_hint": "Hệ thống gợi ý bạn nên đến Phòng 100 (Khám Da liễu Tổng quát) để được bác sĩ sàng lọc ban đầu.",
    })

# ── Alias tiếng Việt để nhận ra bệnh trong câu user (tầng hội thoại) ──
# Đây là CÁCH DUY NHẤT truy cập leprosy/KP/impetigo: user tự nhắc tên/triệu chứng.
DISEASE_ALIASES = {
    "acne":              ["mụn", "mụn trứng cá", "trứng cá", "acne", "mụn bọc", "mụn đầu đen"],
    "atopic_dermatitis": ["viêm da cơ địa", "chàm", "eczema", "chàm cơ địa", "viêm da"],
    "psoriasis":         ["vảy nến", "vẩy nến", "psoriasis"],
    "tinea":             ["nấm da", "hắc lào", "lác", "nấm", "lang ben", "nấm móng", "nấm chân"],
    "wart":              ["mụn cóc", "hạt cơm", "mụn cơm", "wart", "cóc"],
    "leprosy":           ["bệnh phong", "phong", "hủi", "cùi", "lepra", "mất cảm giác", "tê bì"],
    "keratosis_pilaris": ["dày sừng nang lông", "da gà", "sần da gà", "keratosis pilaris", "sừng nang lông"],
    "impetigo":          ["chốc lở", "chốc", "impetigo", "mụn nước vỡ", "đóng vảy mật ong"],
}

# Bệnh có trong RAG nhưng KHÔNG chẩn đoán được từ ảnh (chỉ hội thoại).
CONVERSATION_ONLY_DISEASES = {"leprosy", "keratosis_pilaris", "impetigo"}

# ══════════════════════════════════════════════════════════════════════
#  CLASSIFIER — NGƯỠNG TIN CẬY (từ analyze_threshold.py, bảng CV)
# ══════════════════════════════════════════════════════════════════════
# wart/acne siết cao (giá trị chẩn đoán thấp, thiệt hại khi sai cao).
# tinea nới thấp (giá trị cao, thiệt hại khi sai thấp).
CONF_THRESHOLD = {
    "acne":              0.75,
    "atopic_dermatitis": 0.75,
    "psoriasis":         0.65,
    "tinea":             0.45,
    "wart":              0.85,
}

# ══════════════════════════════════════════════════════════════════════
#  CLASSIFIER — KHUNG NGẮM + QUALITY GATE + TTA
# ══════════════════════════════════════════════════════════════════════
CROP_BOX = (0.20, 0.20, 0.80, 0.80)   # toạ độ chuẩn hoá 0..1; None = cả khung

BLUR_MIN         = float(os.getenv("BLUR_MIN", "5.0"))
BRIGHTNESS_RANGE = (20, 240)

TTA_HFLIP = True   # PHẢI khớp lúc train (T đã fit với TTA bật)

# ══════════════════════════════════════════════════════════════════════
#  TƯƠNG THÍCH NGƯỢC — các tên mà code rag/ cũ vẫn import
# ══════════════════════════════════════════════════════════════════════
# Bệnh KHÔNG có tài liệu RAG. Rỗng vì cả 8 bệnh đều có chunk trong ChromaDB.
# Giữ lại để response_generator.py / query_rewrite.py không vỡ khi import.
DISEASES_WITHOUT_RAG: set = set()

# main.py cũ dùng YOLO_TO_DISEASE trong _detect_disease_in_text. Bản mới dùng
# DISEASE_ALIASES thay thế, nhưng giữ alias rỗng này để mọi import cũ không vỡ.
YOLO_TO_DISEASE: dict = {}
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"   # không còn dùng, giữ cho import cũ
