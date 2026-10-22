"""
audio/stt.py — Speech-to-Text dùng faster-whisper.

Chọn thiết bị qua biến môi trường:
    export STT_DEVICE=cuda      # GPU (nhanh, nhưng tốn VRAM — coi chừng OOM)
    export STT_DEVICE=cpu       # CPU (mặc định, an toàn)
    export STT_BEAM_SIZE=1      # greedy (nhanh); =5 chính xác hơn nhưng chậm
    export STT_CPU_THREADS=6    # số core CPU (Jetson Orin Nano có 6)

Model đọc từ WHISPER_MODEL (mặc định 'small').
→ PhoWhisper (tiếng Việt tốt hơn): export WHISPER_MODEL=~/phowhisper-small-ct2
"""
import os
from typing import Optional

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ── Thiết bị + compute ────────────────────────────────────────────────────
# cuda nhanh hơn CPU nhiều, nhưng chia sẻ 8GB VRAM với classifier -> rủi ro
# OOM. Nếu GPU lỗi, tự lùi về CPU thay vì crash cả robot.
STT_DEVICE  = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE = os.getenv("STT_COMPUTE", "float16" if STT_DEVICE == "cuda" else "int8")

# beam_size: 1 = greedy (nhanh nhất). 5 = chính xác hơn, chậm ~3-5 lần.
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "1"))

# Số core CPU (chỉ dùng khi device=cpu). Jetson Orin Nano = 6. Chừa 1 cho hệ thống.
CPU_THREADS = int(os.getenv("STT_CPU_THREADS", str(max(1, (os.cpu_count() or 4) - 1))))

INITIAL_PROMPT = (
    "Đây là cuộc tư vấn về bệnh da liễu. Các bệnh gồm: vảy nến, viêm da cơ địa, "
    "bệnh phong, mụn trứng cá, dày sừng nang lông, mụn cóc. Người dùng hỏi về "
    "triệu chứng, biểu hiện, nguyên nhân, cách điều trị, phòng ngừa, biến chứng."
)

_model = None


def _get_model():
    """Load model 1 lần, giữ trong RAM. Chọn device theo env, fallback CPU nếu GPU lỗi."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[STT] Loading '{WHISPER_MODEL}' ({STT_DEVICE}/{STT_COMPUTE}, "
              f"beam={BEAM_SIZE}"
              + (f", {CPU_THREADS} threads" if STT_DEVICE == "cpu" else "")
              + ")...", flush=True)

        kwargs = dict(
            model_size_or_path = WHISPER_MODEL,
            device             = STT_DEVICE,
            compute_type       = STT_COMPUTE,
        )
        if STT_DEVICE == "cpu":
            kwargs["cpu_threads"] = CPU_THREADS

        try:
            _model = WhisperModel(**kwargs)
        except Exception as e:
            if STT_DEVICE == "cuda":
                print(f"[STT] ⚠️  GPU lỗi ({e})", flush=True)
                print(f"[STT] ⚠️  Lùi về CPU int8 — kiểm tra VRAM bằng tegrastats", flush=True)
                _model = WhisperModel(
                    model_size_or_path=WHISPER_MODEL, device="cpu",
                    compute_type="int8", cpu_threads=CPU_THREADS)
            else:
                raise
        print(f"[STT] ✅ Model ready (device thực tế sẽ hiện ở dòng trên)", flush=True)
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
            beam_size  = BEAM_SIZE,
            initial_prompt             = INITIAL_PROMPT,
            condition_on_previous_text = False,
            # recorder.py ĐÃ cắt im lặng bằng VAD -> tắt vad_filter (tránh chạy 2 lần)
            vad_filter = False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
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
    _get_model()


if __name__ == "__main__":
    from audio.recorder import record_audio
    print("=== Test STT ===")
    preload()
    wav = record_audio()
    if wav:
        print(f"Kết quả: {transcribe(wav)}")
