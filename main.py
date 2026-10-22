"""
main.py — Conversational RAG + Voice I/O + Streaming TTS

Thay đổi so với bản YOLO:
  - YOLODetector -> SkinClassifier (qua classifier_bridge)
  - Xử lý 4 tình huống ảnh: ok / other / low_confidence / ảnh-mờ
  - Lời thoại an toàn "luôn khuyên tái khám" gắn vào mọi chẩn đoán
  - Bệnh chỉ-hội-thoại (phong/KP/chốc) truy cập qua alias, không qua ảnh
"""

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    CLASSIFIER_MODEL_PATH, LLM_BACKEND,
    DISEASE_VI_NAMES, DISEASE_ALIASES,
    CONVERSATION_ONLY_DISEASES,
)
from rag.retriever import Retriever
from rag.query_rewrite import build_search_query
from rag.response_generator import generate_response
from rag.vector_store import get_collection_info


# ── Opening cho tình huống KHÔNG có ảnh / không chẩn đoán ─────────────────────

OPENING_NO_IMAGE = (
    "Xin chào, tôi là trợ lý tư vấn da liễu. "
    "Bạn hãy mô tả triệu chứng: cảm giác ngứa đau rát, "
    "hình dạng mảng vảy mụn, màu sắc, vị trí trên cơ thể."
)

EXIT_KEYWORDS = {"thoát", "thoat", "exit", "quit", "bye", "q", "dừng", "kết thúc"}


@dataclass
class ConversationSession:
    image_path:       Optional[str]
    detected_disease: Optional[str]
    yolo_confidence:  Optional[float]
    opening_text:     str = ""                      # opening đã soạn từ bridge
    history:          List[Dict] = field(default_factory=list)
    context_disease:  Optional[str] = None
    session_id:       str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def active_disease(self) -> Optional[str]:
        return self.context_disease or self.detected_disease

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def recent_history(self, n: int = 4) -> List[Dict]:
        return self.history[-(n * 2):]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def get_user_input(voice_mode: bool) -> Optional[str]:
    if voice_mode:
        from audio.recorder import record_audio
        from audio.stt import transcribe
        wav = record_audio()
        if wav is None:
            return None
        return transcribe(wav)
    else:
        try:
            return input("👤  ").strip()
        except (KeyboardInterrupt, EOFError):
            return None


def speak_opening(text: str, voice_mode: bool):
    print(f"\n🤖  {text}\n", flush=True)
    if voice_mode:
        from audio.tts import speak_bg
        t = speak_bg(text)
        t.join()


def _detect_disease_in_text(text: str) -> Optional[str]:
    """Nhận ra bệnh trong câu user qua alias. CÁCH DUY NHẤT tới phong/KP/chốc."""
    lower = text.lower()
    for disease_key, aliases in DISEASE_ALIASES.items():
        if any(a in lower for a in aliases):
            return disease_key
    return None


def check_rag_ready() -> bool:
    info = get_collection_info()
    if info["status"] != "ok" or info["count"] == 0:
        print("❌ ChromaDB chưa build. Chạy: python build_rag_full.py")
        return False
    print(f"[Main] ChromaDB OK — {info['count']} chunks")
    return True


def run_classifier(image_path: str):
    """
    Chạy classifier trên ảnh -> ChanDoanResult.
    Trả về (disease_key | None, confidence | None, opening_text).
    disease_key chỉ khác None khi có chẩn đoán đủ tự tin (reason='ok').
    """
    if not Path(CLASSIFIER_MODEL_PATH).exists():
        print(f"⚠️  Không tìm thấy classifier model: {CLASSIFIER_MODEL_PATH}")
        return None, None, None

    from classifier_bridge import ClassifierBridge
    bridge = ClassifierBridge()
    result = bridge.run(image_path)

    # Log cho người vận hành
    print(f"[CLS] reason={result.reason}", end="")
    if result.has_diagnosis:
        vi = DISEASE_VI_NAMES.get(result.disease_key, result.disease_key)
        print(f" -> {vi} ({result.confidence:.0%})")
    else:
        print(" -> không chẩn đoán từ ảnh")
    if result.top3:
        for k, v, p in result.top3[:3]:
            print(f"       • {v}: {p:.0%}")

    key = result.disease_key if result.has_diagnosis else None
    return key, result.confidence, result.opening


# ── Conversation loop ─────────────────────────────────────────────────────────

def start_conversation(session, retriever, voice_mode=False, use_llm_rewrite=False):
    mode_icon = "🎤" if voice_mode else "⌨️ "
    print(f"\n{'═' * 60}")
    print(f"  🏥  HỆ THỐNG TƯ VẤN DA LIỄU  [{mode_icon} #{session.session_id}]")
    print(f"{'═' * 60}")

    # Opening: ưu tiên opening đã soạn từ bridge; nếu không có ảnh -> lời mặc định
    opening = session.opening_text or OPENING_NO_IMAGE
    if not voice_mode:
        opening += "\n(Gõ 'thoát' để kết thúc)"

    speak_opening(opening, voice_mode)
    session.add_turn("assistant", opening)

    while True:
        user_input = get_user_input(voice_mode)

        if user_input is None:
            farewell = "Kết thúc phiên. Chúc bạn sức khoẻ!"
            print(f"\n🤖  {farewell}")
            if voice_mode:
                from audio.tts import speak
                speak(farewell)
            break

        if not user_input:
            if voice_mode:
                print("   ⚠️  Không nghe rõ, thử lại.")
            continue

        if user_input.lower().strip() in EXIT_KEYWORDS:
            farewell = "Cảm ơn bạn đã sử dụng hệ thống. Chúc bạn sức khoẻ!"
            print(f"\n🤖  {farewell}")
            if voice_mode:
                from audio.tts import speak
                speak(farewell)
            break

        # Nhận ra bệnh trong câu user (gồm cả phong/KP/chốc qua alias)
        mentioned = _detect_disease_in_text(user_input)
        if mentioned and mentioned != session.active_disease:
            session.context_disease = mentioned
            vi = DISEASE_VI_NAMES.get(mentioned, mentioned)
            note = " (tư vấn qua mô tả — bệnh này cần bác sĩ khám để chẩn đoán)" \
                   if mentioned in CONVERSATION_ONLY_DISEASES else ""
            print(f"   💡 Nhận ra: {vi}{note}")

        search_query = build_search_query(
            user_query=user_input, disease_key=session.active_disease,
            use_llm=use_llm_rewrite, history=session.recent_history(3),
        )
        print(f"   🔍 [{search_query}]")

        chunks = retriever.query(
            query_text=search_query, disease_key=session.active_disease,
        )
        diseases_found = ", ".join(sorted(set(c["disease_vi"] for c in chunks)))
        print(f"   📄 {len(chunks)} chunks — {diseases_found}\n")

        session.add_turn("user", user_input)

        response = generate_response(
            user_query=user_input, retrieved_chunks=chunks,
            disease_key=session.active_disease, yolo_confidence=session.yolo_confidence,
            history=session.recent_history(4)[:-1], voice_mode=voice_mode,
        )

        session.add_turn("assistant", response)
        print()


def main():
    parser = argparse.ArgumentParser(description="Skin Disease Conversational RAG")
    parser.add_argument("--image")                                  # KHÔNG còn required
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--no-classify", action="store_true", help="Bỏ qua ảnh, chỉ hội thoại")
    parser.add_argument("--llm-rewrite", action="store_true")
    args = parser.parse_args()

    if args.image and not Path(args.image).exists():
        print(f"❌ Không tìm thấy ảnh: {args.image}")
        sys.exit(1)

    if not check_rag_ready():
        sys.exit(1)

    if args.voice:
        print("[Voice] Pre-loading STT model...")
        from audio.stt import preload
        preload()
        import subprocess
        if subprocess.run(["which", "ffplay"], capture_output=True).returncode != 0:
            print("⚠️  ffplay chưa có. Cài: sudo apt install -y ffmpeg")

    disease_key, confidence, opening_text = None, None, None
    if args.image and not args.no_classify:
        disease_key, confidence, opening_text = run_classifier(args.image)

    session = ConversationSession(
        image_path=args.image, detected_disease=disease_key,
        yolo_confidence=confidence, opening_text=opening_text or "",
    )
    retriever = Retriever()

    start_conversation(session, retriever,
                       voice_mode=args.voice, use_llm_rewrite=args.llm_rewrite)


if __name__ == "__main__":
    main()
