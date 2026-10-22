"""
audio/tts.py — Text-to-Speech dùng edge-tts (Microsoft, online, miễn phí).

Giọng Việt có sẵn:
  vi-VN-HoaiMyNeural  — nữ (mặc định, tự nhiên)
  vi-VN-NamMinhNeural — nam

Playback qua ffplay (ffmpeg). Cài: sudo apt install -y ffmpeg
Cài edge-tts: pip install edge-tts
"""

import asyncio
import os
import subprocess
import tempfile
import threading
from typing import Optional

VOICE     = "vi-VN-HoaiMyNeural"   # đổi thành NamMinhNeural nếu muốn giọng nam
RATE      = "+0%"                   # tốc độ đọc: "+10%" nhanh hơn, "-10%" chậm hơn
VOLUME    = "+0%"                   # âm lượng

# Player thử theo thứ tự
_PLAYERS = [
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpg123", "-q"],
    ["mplayer", "-really-quiet"],
]


def _check_player() -> Optional[list]:
    """Tìm audio player khả dụng trên hệ thống."""
    for cmd in _PLAYERS:
        try:
            r = subprocess.run(
                ["which", cmd[0]],
                capture_output=True
            )
            if r.returncode == 0:
                return cmd
        except Exception:
            continue
    return None


async def _generate_audio_async(text: str, out_path: str):
    """Tạo file MP3 từ text dùng edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, volume=VOLUME)
    await communicate.save(out_path)


def _clean_text_for_tts(text: str) -> str:
    """Bỏ markdown formatting trước khi đọc."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)         # italic
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'^\s*[•\-\*]\s+', '', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'\[(\d+)\.\s[^\]]+\]', '', text)  # RAG source tags
    text = re.sub(r'\n{2,}', '. ', text)              # paragraph breaks → pause
    return text.strip()


def speak(text: str) -> bool:
    """
    Đọc text qua loa bluetooth.

    Returns:
        True nếu thành công, False nếu lỗi.
    """
    if not text or not text.strip():
        return True

    player = _check_player()
    if player is None:
        print("[TTS] ❌ Không tìm thấy audio player.", flush=True)
        print("      Cài: sudo apt install -y ffmpeg", flush=True)
        return False

    clean_text = _clean_text_for_tts(text)
    if not clean_text:
        return True

    tmp_path = None
    try:
        # Tạo file MP3 tạm
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        # Tạo audio — dùng event loop mới tránh xung đột
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_generate_audio_async(clean_text, tmp_path))
        finally:
            loop.close()

        # Phát audio qua loa
        subprocess.run(
            player + [tmp_path],
            check          = True,
            capture_output = True,
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"[TTS] Lỗi player: {e}", flush=True)
        return False
    except Exception as e:
        print(f"[TTS] Lỗi: {e}", flush=True)
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
