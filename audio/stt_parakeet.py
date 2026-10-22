"""
audio/stt_parakeet.py — STT dùng Parakeet CTC ONNX trên GPU.

CÙNG INTERFACE với audio/stt.py cũ:
    transcribe(audio_path) -> text | None
    preload()

Thay PhoWhisper: đổi import trong robot_main.py
    from audio.stt import transcribe, preload
 -> from audio.stt_parakeet import transcribe, preload

Hiệu năng: RTF ~0.016 trên GPU (nhanh gấp ~35 lần PhoWhisper CPU).

Cấu hình qua env:
    PARAKEET_ONNX_DIR   thư mục chứa model.onnx + tokenizer.model (mặc định ~/parakeet_onnx)
    PARAKEET_EP         cuda (mặc định) / tensorrt / cpu
"""
import os
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import numpy as np

_REPO_PARAKEET = Path(__file__).resolve().parent.parent / "models" / "parakeet"
_DEFAULT_DIR   = str(_REPO_PARAKEET) if (_REPO_PARAKEET / "model.onnx").exists() else os.path.expanduser("~/parakeet_onnx")

MODEL_DIR = os.getenv("PARAKEET_ONNX_DIR", _DEFAULT_DIR)
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
SPM_PATH  = os.path.join(MODEL_DIR, "tokenizer.model")
EP        = os.getenv("PARAKEET_EP", "cpu" if os.name == "nt" else "cuda")

# Tham số feature extraction — KHỚP NeMo config
SR         = 16000
N_FFT      = 512
WIN_LENGTH = int(0.025 * SR)   # 400
HOP_LENGTH = int(0.01 * SR)    # 160
N_MELS     = 80
DITHER     = 1e-05

_sess = None
_sp = None
_use_groq = False


def _get_session():
    """Load ONNX session + tokenizer 1 lần. Nếu lỗi, chuyển sang Groq Whisper."""
    global _sess, _sp, _use_groq
    if _use_groq:
        return None, None

    if _sess is None:
        try:
            import onnxruntime as ort
            import sentencepiece as spm

            if not os.path.exists(ONNX_PATH) or not os.path.exists(SPM_PATH):
                raise FileNotFoundError(f"Không thấy model file tại {MODEL_DIR}")

            ep_map = {
                "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                "cpu":      ["CPUExecutionProvider"],
            }
            providers = ep_map.get(EP, ep_map["cuda"])
            print(f"[STT] Parakeet ONNX — provider: {providers[0]}", flush=True)
            try:
                _sess = ort.InferenceSession(ONNX_PATH, providers=providers)
            except Exception as e:
                err_text = str(e)
                if "External data" in err_text or "encoder.pre_encode" in err_text:
                    raise  # Đi thẳng xuống fallback Groq Whisper
                # GPU lỗi/OOM -> lùi CPU thay vì crash
                print(f"[STT] [Warn] GPU không khả dụng ({err_text[:40]}) — chuyển CPU...", flush=True)
                _sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

            _sp = spm.SentencePieceProcessor()
            _sp.load(SPM_PATH)
            print(f"[STT] [OK] Parakeet ready ({_sp.get_piece_size()} token)", flush=True)
        except Exception as e:
            err_str = str(e)
            if "encoder.pre_encode" in err_str or "External data" in err_str:
                short_err = "thiếu weights cục bộ (chỉ cài trên Jetson GPU)"
            elif "Không thấy model file" in err_str:
                short_err = "chưa có file model"
            else:
                short_err = err_str[:80]
            print(f"[STT] [Info] Parakeet ONNX: {short_err}.", flush=True)
            print("[STT] [Groq] Tự động kích hoạt Groq Whisper API (whisper-large-v3 — siêu tốc <0.4s).", flush=True)
            _sess = False
            _use_groq = True
            return None, None

    return _sess, _sp


def _extract_features(y):
    """Mel-spectrogram khớp NeMo AudioToMelSpectrogramPreprocessor."""
    import librosa
    if DITHER > 0:
        y = y + DITHER * np.random.randn(*y.shape).astype(y.dtype)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        window="hann", n_mels=N_MELS, power=2.0, center=True, fmin=0.0, fmax=SR/2.0,
    )
    logmel = np.log(mel + (2.0 ** -24))
    mean = logmel.mean(axis=1, keepdims=True)
    std  = logmel.std(axis=1, keepdims=True)
    logmel = (logmel - mean) / (std + 1e-5)
    return logmel.astype(np.float32)


def _ctc_decode(logprobs, blank_id):
    """
    CTC greedy decode CHUẨN:
      - Gộp token lặp LIÊN TIẾP (cùng frame liền nhau) thành 1
      - blank RESET bộ đếm lặp (cho phép token thật lặp lại nếu có blank giữa)
      - Bỏ blank khỏi kết quả
    Khác bản cũ: dùng biến 'prev' reset về blank sau mỗi blank, để token
    đầu từ sau không bị nuốt nhầm khi trùng token cuối từ trước.
    """
    ids = logprobs.argmax(axis=-1)
    tokens = []
    prev = blank_id   # khởi tạo = blank, KHÔNG phải -1
    for i in ids:
        i = int(i)
        if i == blank_id:
            prev = blank_id      # blank reset -> token sau (dù trùng) sẽ được giữ
            continue
        if i != prev:            # chỉ bỏ nếu trùng token NGAY TRƯỚC (chưa qua blank)
            tokens.append(i)
        prev = i
    return tokens


def _transcribe_groq(audio_path: str, language: str = "vi"):
    """Fallback sang Groq Whisper API khi không có GPU Jetson hoặc thiếu weights ONNX."""
    try:
        from groq import Groq
        from config import GROQ_API_KEY
        if not GROQ_API_KEY:
            print("[STT] [Error] Chưa cấu hình GROQ_API_KEY.", flush=True)
            return None
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3",
                language=language,
                response_format="text",
            )
        text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
        text = text.strip() if text else ""
        # Lọc hallucination của Whisper khi khoảng lặng
        _HALLUCINATIONS = (
            "subscribe", "đăng ký kênh", "hãy like", "theo dõi kênh",
            "cảm ơn các bạn đã xem", "cảm ơn các bạn đã theo dõi",
            "hẹn gặp lại các bạn", "la la school", "ghiền mì gõ",
            "để không bỏ lỡ", "nhấn chuông", "chúc các bạn", "tạm biệt các bạn",
        )
        low = text.lower()
        if any(h in low for h in _HALLUCINATIONS) and len(text.split()) < 15:
            return None
        return text if text else None
    except Exception as e:
        print(f"[STT] Groq Whisper API lỗi: {e}", flush=True)
        return None


def transcribe(audio_path: str, language: str = "vi"):
    """Chuyển WAV (16kHz mono) → text. None nếu lặng/lỗi. Cùng interface stt.py."""
    if not audio_path or not os.path.exists(audio_path):
        return None

    text = None
    try:
        sess, sp = _get_session()
        if sess and sp:
            import librosa
            y, _ = librosa.load(audio_path, sr=SR, mono=True)
            if len(y) < SR * 0.2:   # < 0.2s -> coi như im lặng
                return None

            feat = _extract_features(y)                 # [80, T]
            feed = {sess.get_inputs()[0].name: feat[np.newaxis, :, :]}
            if len(sess.get_inputs()) > 1:
                feed[sess.get_inputs()[1].name] = np.array([feat.shape[1]], dtype=np.int64)

            outputs = sess.run(None, feed)
            logprobs = outputs[0]
            if logprobs.ndim == 3:
                logprobs = logprobs[0]
            blank_id = logprobs.shape[-1] - 1

            tokens = _ctc_decode(logprobs, blank_id)
            text = sp.decode(tokens).strip()
        else:
            # Dùng Groq Whisper API
            text = _transcribe_groq(audio_path, language)
    except Exception as e:
        # Nếu Parakeet lỗi runtime, thử Groq Whisper
        text = _transcribe_groq(audio_path, language)
        if text is None:
            print(f"[STT] Lỗi: {e}", flush=True)
            return None
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    if not text:
        print("[STT] [Info] Không nhận ra giọng nói.", flush=True)
        return None
    print(f"[STT] [Text] \"{text}\"", flush=True)
    return text


def preload():
    """Load model vào bộ nhớ (gọi lúc boot để tránh lag lần đầu)."""
    _get_session()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        preload()
        print("Kết quả:", transcribe(sys.argv[1]))
    else:
        from audio.recorder import record_audio
        preload()
        wav = record_audio()
        if wav:
            print("Kết quả:", transcribe(wav))
