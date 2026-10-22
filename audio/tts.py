"""
audio/tts.py — Text-to-Speech hỗ trợ mô hình Piper TTS (cục bộ siêu tốc ~0.12s)
và fallback Microsoft Edge-TTS (đám mây).

Chế độ mặc định: Piper TTS (100% Offline, on-device, RTF < 0.05x).
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import threading
import time
import sys
from typing import Optional, Tuple
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from config import (
        TTS_BACKEND as CFG_TTS_BACKEND,
        ELEVENLABS_API_KEY as CFG_EL_KEY,
        ELEVENLABS_VOICE_ID as CFG_EL_VOICE,
        ELEVENLABS_MODEL_ID as CFG_EL_MODEL,
    )
except Exception:
    CFG_TTS_BACKEND = "elevenlabs"
    CFG_EL_KEY = "sk_432f504677b0d88518feebe9550810fa8fa0615e55b882c2"
    CFG_EL_VOICE = "EXAVITQu4vr4xnSDxMaL"
    CFG_EL_MODEL = "eleven_flash_v2_5"

# Cấu hình backend TTS: "elevenlabs" (chính) -> fallback "piper" (offline) -> "edge"
TTS_BACKEND = os.environ.get("TTS_BACKEND", CFG_TTS_BACKEND).lower()
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", CFG_EL_KEY)
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", CFG_EL_VOICE)
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", CFG_EL_MODEL)

# Đường dẫn model Piper ONNX tiếng Việt (dùng khi chạy Offline)
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "piper")
PIPER_MODEL_PATH = os.environ.get(
    "PIPER_MODEL_PATH",
    os.path.join(_DEFAULT_MODEL_DIR, "banmai.onnx")
)

VOICE     = "vi-VN-HoaiMyNeural"   # HoaiMyNeural tự nhiên, ổn định nhất
RATE      = "+0%"                   # tốc độ đọc: "+10%" nhanh hơn, "-10%" chậm hơn
VOLUME    = "+0%"                   # âm lượng

# Quản lý tiến trình phát âm thanh để ngắt tức thì khi user bấm nút Kết thúc
_active_proc: Optional[subprocess.Popen] = None
_proc_lock = threading.Lock()

# Quản lý instance PiperVoice (singleton)
_piper_voice = None
_piper_lock = threading.Lock()

# Player thử theo thứ tự
_PLAYERS = [
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpg123", "-q"],
    ["mplayer", "-really-quiet"],
]


def _check_player() -> Optional[list]:
    """Tìm audio player khả dụng trên hệ thống."""
    for cmd in _PLAYERS:
        if shutil.which(cmd[0]):
            return cmd
    return None


def stop_audio():
    """Dừng phát âm thanh ngay lập tức (cả sounddevice lẫn ffplay/subprocess)."""
    global _active_proc
    with _proc_lock:
        if _active_proc and _active_proc.poll() is None:
            try:
                _active_proc.terminate()
                _active_proc.wait(timeout=0.2)
            except Exception:
                try:
                    _active_proc.kill()
                except Exception:
                    pass
            _active_proc = None
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass


def play_audio(file_path: str, stop_event: Optional[threading.Event] = None) -> bool:
    """
    Phát file âm thanh (MP3 / WAV) có hỗ trợ dừng tức thì khi stop_event được kích hoạt.
    1. Thử player hệ thống (ffplay, mpg123, mplayer) nếu có.
    2. Fallback sang soundfile + sounddevice (chạy native Windows/Linux/Mac).
    """
    global _active_proc
    if not file_path or not os.path.exists(file_path):
        return False
    if stop_event and stop_event.is_set():
        return False

    player = _check_player()
    if player:
        try:
            with _proc_lock:
                _active_proc = subprocess.Popen(
                    player + [file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            proc = _active_proc
            while proc and proc.poll() is None:
                if stop_event and stop_event.is_set():
                    stop_audio()
                    return False
                time.sleep(0.03)
            with _proc_lock:
                if _active_proc == proc:
                    _active_proc = None
            return proc.returncode == 0
        except Exception:
            pass

    # Fallback native python audio player
    try:
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(file_path)
        sd.play(data, sr)
        while sd.get_stream() and sd.get_stream().active:
            if stop_event and stop_event.is_set():
                sd.stop()
                return False
            time.sleep(0.02)
        return True
    except Exception as e:
        print(f"[TTS] Lỗi phát âm thanh: {e}", flush=True)
        return False


def play_audio_array(data: np.ndarray, sample_rate: int, stop_event: Optional[threading.Event] = None) -> bool:
    """
    Phát trực tiếp mảng audio PCM (numpy.ndarray float32) qua sounddevice trên RAM.
    Không ghi/đọc file đĩa -> loại bỏ hoàn toàn độ trễ I/O.
    Hỗ trợ ngắt tức thì qua stop_event.
    """
    if data is None or len(data) == 0:
        return True
    if stop_event and stop_event.is_set():
        return False
    try:
        import sounddevice as sd
        sd.play(data, sample_rate)
        while sd.get_stream() and sd.get_stream().active:
            if stop_event and stop_event.is_set():
                sd.stop()
                return False
            time.sleep(0.02)
        time.sleep(0.08)  # Đợi phần cứng DAC giải phóng hoàn toàn buffer trước khi chuyển state
        return True
    except Exception as e:
        print(f"[TTS play_audio_array] Lỗi: {e}", flush=True)
        return False


def get_piper_voice():
    """Lấy hoặc nạp mô hình PiperVoice (singleton) kèm warm-up."""
    global _piper_voice
    with _piper_lock:
        if _piper_voice is not None:
            return _piper_voice
        try:
            from piper import PiperVoice
            model_path = os.path.abspath(PIPER_MODEL_PATH)
            if not os.path.exists(model_path):
                # Thử tìm tương đối từ cwd hoặc thư mục models
                for alt in [
                    "models/piper/banmai.onnx",
                    "models/piper/vi_VN-vais1000-medium.onnx",
                    "models/piper/vi_VN-25hours_single-low.onnx",
                    "models/piper/vi_VN-vivos-x_low.onnx",
                ]:
                    if os.path.exists(alt):
                        model_path = os.path.abspath(alt)
                        break

            if not os.path.exists(model_path):
                print(f"[TTS] Khong tim thay file model Piper tai: {model_path}", flush=True)
                return None

            _piper_voice = PiperVoice.load(model_path)
            # Warm-up để lần đầu tiên gọi không bị độ trễ khởi tạo ONNX
            try:
                list(_piper_voice.synthesize("alo"))
            except Exception:
                pass
            print(f"[TTS] Da nap thanh cong mo hinh Piper TTS: {os.path.basename(model_path)}", flush=True)
            return _piper_voice
        except Exception as e:
            print(f"[TTS] Khong the nap Piper TTS ({e}), se dung Edge-TTS.", flush=True)
            return None


def synthesize_pcm(text: str) -> Optional[Tuple[np.ndarray, int]]:
    """
    Sinh mảng audio PCM (numpy.ndarray float32) trực tiếp trên RAM bằng Piper TTS.
    Tốc độ cực nhanh: ~0.08 - 0.12s / câu.
    Bổ sung: Fade-in/out chống nổ tiếng (pop/click), chuẩn hóa âm lượng chống rè/xé tiếng,
    và chèn khoảng im lặng đầu/cuối chống nuốt chữ trên loa Bluetooth/USB.
    """
    voice = get_piper_voice()
    if voice is None:
        return None
    try:
        # Chặn warning C++ phoneme của espeak vào stderr
        f_null = open(os.devnull, 'w')
        old_err = os.dup(2)
        try:
            os.dup2(f_null.fileno(), 2)
            chunks = list(voice.synthesize(text))
        finally:
            try:
                os.dup2(old_err, 2)
                os.close(old_err)
                f_null.close()
            except Exception:
                pass

        if not chunks:
            return None
        audio_data = np.concatenate([c.audio_float_array for c in chunks])
        sr = chunks[0].sample_rate

        # 1. Peak Normalization: Chuẩn hóa đỉnh âm lượng về 0.90 (chống rè/méo tiếng)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.90:
            audio_data = audio_data * (0.90 / max_val)

        # 2. Fade-in (12ms) & Fade-out (18ms): Xóa triệt để tiếng nổ 'tách/bụp' đầu và cuối câu
        fi_len = min(len(audio_data), int(0.012 * sr))
        fo_len = min(len(audio_data), int(0.018 * sr))
        if fi_len > 0:
            audio_data[:fi_len] *= np.linspace(0.0, 1.0, fi_len, dtype=np.float32)
        if fo_len > 0:
            audio_data[-fo_len:] *= np.linspace(1.0, 0.0, fo_len, dtype=np.float32)

        # 3. Chèn im lặng 150ms đầu (chống nuốt chữ đầu) và 250ms cuối (chống nuốt chữ cuối)
        p_start = np.zeros(int(0.15 * sr), dtype=np.float32)
        p_end = np.zeros(int(0.25 * sr), dtype=np.float32)

        final_pcm = np.concatenate([p_start, audio_data, p_end])
        return final_pcm, sr
    except Exception as e:
        print(f"[TTS Piper synthesize] Lỗi: {e}", flush=True)
        return None


def synthesize_elevenlabs(text: str, out_path: str, stop_event: Optional[threading.Event] = None) -> bool:
    """
    Sinh file MP3 dùng ElevenLabs Flash v2.5 siêu tốc (~0.7s) và giọng đọc đỉnh cao thế giới.
    Tự động xử lý tiếng Anh, tên thuốc y khoa và thuật ngữ chuyên ngành.
    """
    if not ELEVENLABS_API_KEY:
        return False
    if stop_event and stop_event.is_set():
        return False

    import urllib.request, urllib.error, json
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=mp3_44100_128"
    payload = json.dumps({
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "User-Agent": "ThinkMedRobot/1.0",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return False
            data = bytearray()
            while True:
                if stop_event and stop_event.is_set():
                    return False
                chunk = resp.read(4096)
                if not chunk:
                    break
                data.extend(chunk)

            if not data:
                return False
            with open(out_path, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"[TTS ElevenLabs] Loi ({e}), se chuyen sang fallback Piper offline.", flush=True)
        return False


async def _generate_audio_async(text: str, out_path: str, stop_event: Optional[threading.Event] = None):
    """Tạo file MP3 từ text dùng edge-tts (có retry, fallback giọng nam, và ngắt tức thì khi stop_event set)."""
    import edge_tts
    import re

    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text.strip()] if text.strip() else []
    if not raw_sentences:
        return

    sentences = []
    for s in raw_sentences:
        if not s.endswith(('.', '!', '?', ':')):
            s = s + '.'
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
        sentences.append(s)

    combined_data = bytearray()
    voices_to_try = [VOICE, "vi-VN-NamMinhNeural"]

    for s in sentences:
        if stop_event and stop_event.is_set():
            return
        sent_data = bytearray()
        for v in voices_to_try:
            for attempt in range(2):
                if stop_event and stop_event.is_set():
                    return
                try:
                    communicate = edge_tts.Communicate(s, v, rate=RATE, volume=VOLUME)
                    async for chunk in communicate.stream():
                        if stop_event and stop_event.is_set():
                            return
                        if chunk["type"] == "audio":
                            sent_data.extend(chunk["data"])
                    if sent_data:
                        break
                except Exception:
                    await asyncio.sleep(0.15)
            if sent_data:
                break
        if sent_data:
            combined_data.extend(sent_data)

    if not combined_data:
        raise RuntimeError("No audio data generated from TTS service.")

    with open(out_path, "wb") as f:
        f.write(combined_data)


# Danh mục phiên âm các thuật ngữ thuốc / y khoa tiếng Anh sang tiếng Việt chuẩn
_MED_PHONETICS = [
    (r'\bcorticosteroids?\b', 'coọc ti cô xte rô it'),
    (r'\bmethotrexate\b', 'mê tô tơ rê xát'),
    (r'\bisotretinoin\b', 'i sô trê ti nô in'),
    (r'\bparacetamol\b', 'pa ra xê ta môn'),
    (r'\bantibiotics?\b', 'kháng sinh'),
    (r'\blasers?\b', 'la-ze'),
    (r'\bbiops(y|ies)\b', 'sinh thiết'),
    (r'\bkeratolysis\b', 'tiêu sừng'),
    (r'\bbenzoyl peroxide\b', 'ben dôi pơ rốc xít'),
    (r'\bclindamycin\b', 'clin đa mi xin'),
    (r'\berythromycin\b', 'ê ri thô mi xin'),
    (r'\bhydrocortisone\b', 'hy đrô coọc ti xôn'),
    (r'\btretinoin\b', 'trê ti nô in'),
    (r'\badapalene\b', 'a đa pa len'),
    (r'\bketoconazole\b', 'kê tô cô na dôn'),
    (r'\bitraconazole\b', 'i tra cô na dôn'),
    (r'\bfluconazole\b', 'phlu cô na dôn'),
    (r'\bterbinafine\b', 'téc bi na phin'),
    (r'\bsalicylic acid\b', 'a xít sa li xi líc'),
    (r'\bcorticoid\b', 'coọc ti cô it'),
    (r'\beczema\b', 'ếch dê ma'),
    (r'\bpsoriasis\b', 'vẩy nến'),
    (r'\bacne\b', 'mụn trứng cá'),
    (r'\bwart\b', 'mụn cóc'),
    (r'\bnấm dermatophytes?\b', 'nấm sợi'),
    (r'\bdermatophytes?\b', 'nấm sợi'),
    (r'\btinea\b', 'nấm da'),
    (r'\bviruses?\b', 'vi-rút'),
    (r'\bDNA\b', 'Đê-En-A'),
    (r'\bRNA\b', 'A-Er-En'),
    (r'\bAPI\b', 'A-P-I'),
    (r'\bYOLO\b', 'Yo-lo'),
]


def _clean_text_for_tts(text: str) -> str:
    """Bỏ markdown formatting và chuẩn hóa phiên âm từ ngữ y tế trước khi đọc."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)         # italic
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'^\s*[•\-\*]\s+', '', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'\[(\d+)\.\s[^\]]+\]', '', text)  # RAG source tags
    text = re.sub(r'\n{2,}', '. ', text)              # paragraph breaks → pause

    # Chuẩn hóa từ ngữ y tế & tên thương hiệu tiếng Việt chống đọc nhầm từ thô tục
    text = re.sub(r'\bThinkMed\b', 'Thinh Mét', text, flags=re.IGNORECASE)
    text = re.sub(r'\bThink\s+Med\b', 'Thinh Mét', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAI\b', 'A-I', text)
    text = re.sub(r'\bmạn tính\b', 'mãn tính', text, flags=re.IGNORECASE)
    text = re.sub(r'\bHPV\b', 'vi-rút H-P-V', text)
    text = re.sub(r'\bHIV\b', 'H-I-V', text)

    # Phiên âm các thuật ngữ thuốc / y khoa tiếng Anh sang tiếng Việt tự nhiên
    for pattern, replacement in _MED_PHONETICS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text.strip()


def _pad_silence(mp3_path: str, ms: int = 150) -> str:
    """
    Chèn `ms` mili-giây im lặng vào ĐẦU file MP3 để loa Bluetooth không nuốt
    chữ đầu (link A2DP cần vài trăm ms để 'vào nhịp' khi bắt đầu phát).
    Trả về đường dẫn file đã chèn; nếu ms <= 0 hoặc ffmpeg lỗi thì trả về file gốc.
    """
    if ms <= 0:
        return mp3_path

    out = mp3_path[:-4] + "_pad.mp3" if mp3_path.endswith(".mp3") else mp3_path + "_pad.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-af", f"adelay={ms}:all=1", out],
            check=True, capture_output=True,
        )
        return out
    except Exception:
        return mp3_path   # không chèn được thì phát file gốc, vẫn chạy


def _speak_edge(clean_text: str, stop_event: Optional[threading.Event] = None) -> bool:
    """Đọc text qua Edge-TTS (fallback đám mây)."""
    tmp_path = None
    try:
        # Tạo file MP3 tạm
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        if stop_event and stop_event.is_set():
            return False

        # Tạo audio — dùng event loop mới tránh xung đột
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_generate_audio_async(clean_text, tmp_path, stop_event=stop_event))
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

        if stop_event and stop_event.is_set():
            return False

        # Chèn im lặng đầu file rồi phát qua loa (chống nuốt chữ đầu loa bluetooth)
        play_path = _pad_silence(tmp_path)
        try:
            return play_audio(play_path, stop_event=stop_event)
        finally:
            if play_path != tmp_path and os.path.exists(play_path):
                try:
                    os.unlink(play_path)
                except Exception:
                    pass
        return True

    except subprocess.CalledProcessError as e:
        print(f"[TTS Edge] Lỗi player: {e}", flush=True)
        return False
    except Exception as e:
        print(f"[TTS Edge] Lỗi: {e}", flush=True)
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def speak(text: str, stop_event: Optional[threading.Event] = None) -> bool:
    """
    Đọc text qua loa có hỗ trợ dừng tức thì qua stop_event.
    1. Thử ElevenLabs Flash v2.5 (siêu tốc ~0.7s, giọng cực hay, chuẩn tiếng Anh y tế).
    2. Fallback sang Piper TTS (offline cục bộ 100%).
    3. Fallback sang Edge-TTS nếu cả 2 gặp sự cố.
    """
    if not text or not text.strip():
        return True
    if stop_event and stop_event.is_set():
        return False

    clean_text = _clean_text_for_tts(text)
    if not clean_text:
        return True

    # 1. Thử ElevenLabs nếu cấu hình là "elevenlabs"
    if TTS_BACKEND == "elevenlabs" and ELEVENLABS_API_KEY:
        tmp_mp3 = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_mp3 = f.name
            print(f"[TTS] [ElevenLabs] Đang phát qua API ({ELEVENLABS_MODEL_ID})...", flush=True)
            ok = synthesize_elevenlabs(clean_text, tmp_mp3, stop_event=stop_event)
            if ok and os.path.exists(tmp_mp3):
                play_path = _pad_silence(tmp_mp3)
                try:
                    return play_audio(play_path, stop_event=stop_event)
                finally:
                    for p in {tmp_mp3, play_path}:
                        if os.path.exists(p):
                            try: os.unlink(p)
                            except Exception: pass
            else:
                print("[TTS] [Warn] ElevenLabs không phản hồi, chuyển sang fallback Piper offline...", flush=True)
        except Exception as e:
            print(f"[TTS] [Warn] Lỗi ElevenLabs ({e}), chuyển sang fallback Piper...", flush=True)
            if tmp_mp3 and os.path.exists(tmp_mp3):
                try: os.unlink(tmp_mp3)
                except Exception: pass

    # 2. Thử Piper TTS cục bộ (chế độ Offline 100% hoặc khi ElevenLabs fallback)
    if TTS_BACKEND == "piper":
        print("[TTS] [Piper] Đang phát bằng Piper TTS (Offline cục bộ)...", flush=True)
        res = synthesize_pcm(clean_text)
        if res is not None:
            data, sr = res
            return play_audio_array(data, sr, stop_event=stop_event)
        print("[TTS] [Warn] Piper không khả dụng, chuyển sang fallback Edge-TTS...", flush=True)

    # 3. Microsoft Edge-TTS API (Chuẩn tiếng Việt phát thanh viên tự nhiên nhất)
    print(f"[TTS] [Edge-TTS] Đang phát bằng Microsoft Edge-TTS API ({VOICE})...", flush=True)
    return _speak_edge(clean_text, stop_event=stop_event)


def speak_bg(text: str) -> threading.Thread:
    """
    Đọc text ở background (non-blocking).
    Trả về Thread để caller có thể join() nếu cần.
    """
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()
    return t


def test_tts(text: str = "Xin chào, hệ thống tư vấn da liễu đang hoạt động.") -> bool:
    """Test TTS — gọi trực tiếp để kiểm tra loa."""
    print(f"[Test TTS] Đọc: \"{text}\"")
    return speak(text)


if __name__ == "__main__":
    test_tts()
