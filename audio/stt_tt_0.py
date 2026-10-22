"""
audio/stt.py — Speech-to-Text dùng faster-whisper.

Model đọc từ biến môi trường WHISPER_MODEL (mặc định 'small').
→ Đổi sang PhoWhisper (tiếng Việt tốt hơn HẲN): export WHISPER_MODEL=~/phowhisper-small-ct2
Compute: CPU int8 (nhẹ VRAM, nhường GPU cho YOLO).
"""
import os
from typing import Optional

# 'small' = whisper đa ngôn ngữ (yếu tiếng Việt).
# Đổi bằng:  export WHISPER_MODEL=/home/fptuadmin/phowhisper-small-ct2
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Mồi từ vựng chuyên ngành để nhận đúng thuật ngữ.
INITIAL_PROMPT = (
    "Đây là cuộc tư vấn về bệnh da liễu. Các bệnh gồm: vảy nến, viêm da cơ địa, "
    "bệnh phong, mụn trứng cá, dày sừng nang lông, mụn cóc. Người dùng hỏi về "
    "triệu chứng, biểu hiện, nguyên nhân, cách điều trị, phòng ngừa, biến chứng."
)

_model = None
def _get_model():
    """Load model 1 lần duy nhất, giữ trong RAM cho các lần sau."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[STT] Loading faster-whisper '{WHISPER_MODEL}' (CPU int8)... "
              f"(lần đầu tải model, chờ chút)", flush=True)
        _model = WhisperModel(
            model_size_or_path = WHISPER_MODEL,
            device             = "gpu",
            compute_type       = "float16",
        )
        print("[STT] ✅ Model ready", flush=True)
    return _model
def transcribe(audio_path: str, language: str = "vi") -> Optional[str]:
    """Chuyển file WAV (16kHz mono) → text. None nếu lặng/lỗi."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    model = _get_model()
    try:
        segments, info = model.transcribe(
            audio_path,
            language   = language,
            beam_size  = 5,
            initial_prompt             = INITIAL_PROMPT,
            condition_on_previous_text = False,
            vad_filter = False,
            vad_parameters = {"min_silence_duration_ms": 500},
        )
        text_parts = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                text_parts.append(t)
        text = " ".join(text_parts).strip()
    except Exception as e:
        print(f"[STT] Lỗi transcribe: {e}", flush=True)
        return None
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass
    if not text:
        print("[STT] ⚠️  Không nhận ra giọng nói.", flush=True)
        return None
    print(f"[STT] 📝 \"{text}\"", flush=True)
    return text
def preload():
    """Pre-load model vào RAM (gọi khi khởi động để tránh lag lần đầu)."""
    _get_model()
if __name__ == "__main__":
    from audio.recorder import record_audio
    print("=== Test STT ===")
    preload()
    wav = record_audio()
    if wav:
        print(f"Kết quả: {transcribe(wav)}")
