"""
audio/recorder.py — VAD-based recording (không cần nhấn nút).

Thay đổi: record_vad() nhận thêm max_wait_sec — cho phép dùng thời gian
chờ ngắn hơn khi hỏi câu check-in "bạn còn muốn hỏi thêm không?"
(mặc định 30s cho câu hỏi mở đầu, ~12s cho câu trả lời có/không).

Cài: pip install sounddevice soundfile numpy soxr
"""

import os
import sys
import tempfile
import time
import threading
from typing import Optional
import numpy as np
import sounddevice as sd
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Tự tìm mic (BỀN với việc index thiết bị đổi sau mỗi lần reboot) ────────────
# Nếu auto-detect chọn sai, ép index cứng qua biến môi trường:  export MIC_DEVICE=3
def _ensure_pulse_mic():
    if os.name != "posix":
        return False
    try:
        import subprocess
        out = subprocess.run(["pactl", "list", "short", "sources"],
                             capture_output=True, text=True, timeout=4).stdout
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                name = parts[1]
                low  = name.lower()
                if "usb" in low and "monitor" not in low:
                    subprocess.run(["pactl", "set-default-source", name], timeout=4)
                    print(f"[Recorder] [OK] Đặt default source = {name}", flush=True)
                    return True
        print("[Recorder] [Warn] Không thấy source USB trong PulseAudio", flush=True)
    except Exception as e:
        print(f"[Recorder] [Warn] Không đặt được default source ({e})", flush=True)
    return False


def _find_mic() -> int:
    _ensure_pulse_mic()          # đảm bảo default source là mic USB trước khi chọn
    override = os.getenv("MIC_DEVICE")
    if override is not None:
        print(f"[Recorder] [Mic] Mic (ép qua env MIC_DEVICE={override})", flush=True)
        return int(override)
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[Recorder] [Warn] Không liệt kê được thiết bị ({e}) — dùng device 0", flush=True)
        return 0
    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) >= 1]
    if not inputs:
        print("[Recorder] [Warn] Không thấy thiết bị THU nào — dùng default", flush=True)
        return None
    # 1) Ưu tiên 'pulse' → đi qua PulseAudio, lấy đúng default source (mic USB).
    #    Trên Jetson này card USB bị PulseAudio chiếm, nên PHẢI thu qua pulse.
    for i, d in inputs:
        if d["name"].strip().lower() == "pulse":
            print(f"[Recorder] [Mic] Mic qua PulseAudio: [{i}] pulse", flush=True)
            return i
    # 2) 'default'
    for i, d in inputs:
        if d["name"].strip().lower() == "default":
            print(f"[Recorder] [Mic] Mic (default): [{i}]", flush=True)
            return i
    # 3) Tên có 'usb' (mic USB xuất hiện trực tiếp)
    for i, d in inputs:
        if "usb" in d["name"].lower():
            print(f"[Recorder] [Mic] Mic USB: [{i}] {d['name']}", flush=True)
            return i
    # 4) Input đầu tiên KHÔNG phải thiết bị nội bộ (APE/tegra/hdmi → mở sẽ treo)
    SKIP = ("ape", "tegra", "hdmi", "dummy", "monitor")
    for i, d in inputs:
        if not any(s in d["name"].lower() for s in SKIP):
            print(f"[Recorder] [Mic] Mic: [{i}] {d['name']}", flush=True)
            return i
    # 5) Cùng lắm mới dùng default (tránh trả về APE gây treo)
    print("[Recorder] [Warn] Chỉ thấy thiết bị nội bộ — dùng default", flush=True)
    return None

MIC_DEVICE      = _find_mic()
NATIVE_RATE     = 48000
WHISPER_RATE    = 16000
CHUNK_SAMPLES   = int(NATIVE_RATE * 0.05)   # 50ms/chunk

SILENCE_THRESHOLD    = float(os.getenv("VAD_THRESHOLD", "0.004"))
MIN_SPEECH_DURATION  = 0.3
MAX_SILENCE_AFTER    = 1.0
MAX_RECORD_DURATION  = 15.0


def _resample(audio: np.ndarray, orig: int, target: int) -> np.ndarray:
    if orig == target:
        return audio
    try:
        import soxr
        return soxr.resample(audio.flatten(), orig, target).reshape(-1, 1)
    except ImportError:
        ratio = target / orig
        n     = int(len(audio) * ratio)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, audio.flatten()).reshape(-1, 1)


def play_chime():
    """Phát tiếng chuông 'Ting' nhẹ êm tai (880Hz -> 1320Hz) báo hiệu mic đã mở."""
    try:
        sr = 24000
        t1 = np.linspace(0, 0.07, int(sr * 0.07), False)
        t2 = np.linspace(0, 0.10, int(sr * 0.10), False)
        tone1 = 0.15 * np.sin(2 * np.pi * 880 * t1) * np.exp(-t1 * 20)
        tone2 = 0.18 * np.sin(2 * np.pi * 1320 * t2) * np.exp(-t2 * 15)
        chime = np.concatenate([tone1, tone2]).astype(np.float32)
        sd.play(chime, sr)
        sd.wait()
    except Exception as e:
        pass


def record_vad(
    on_listening_start=None,
    on_speech_start=None,
    on_speech_end=None,
    on_energy_update=None,
    stop_event: Optional[threading.Event] = None,
    max_wait_sec: float = 30.0,
) -> str | None:
    """
    Ghi âm tự động dùng VAD.
    - Phát tiếng Ting báo hiệu mở mic.
    - Cập nhật năng lượng giọng nói thực tế để UI đổi màu và co giãn sóng âm.
    - Hỗ trợ dừng tức thì khi người dùng chạm nút Kết thúc.
    """
    # Khoảng đệm chống tiếng vang từ loa ngoài (echo guard delay)
    time.sleep(0.35)

    # Phát âm thanh báo hiệu mở mic
    play_chime()

    if on_listening_start:
        on_listening_start()

    speech_frames = []
    in_speech     = False
    speech_start  = None
    last_speech   = None
    done          = threading.Event()
    result_frames = []

    print(f"   [Mic] Đang lắng nghe... (nói vào mic, tối đa {max_wait_sec:.0f}s)", flush=True)

    def _callback(indata, frame_count, time_info, status):
        nonlocal in_speech, speech_start, last_speech

        energy    = float(np.abs(indata).mean())
        is_speech = energy > SILENCE_THRESHOLD

        if on_energy_update:
            try:
                on_energy_update(energy, in_speech or is_speech)
            except Exception:
                pass

        if is_speech:
            if not in_speech:
                in_speech    = True
                speech_start = time.time()
                print("   [Mic] Đã nhận diện giọng nói, đang thu...", flush=True)
                if on_speech_start:
                    on_speech_start()
            last_speech = time.time()
            speech_frames.append(indata.copy())
        else:
            if in_speech:
                speech_frames.append(indata.copy())
                if last_speech and time.time() - last_speech > MAX_SILENCE_AFTER:
                    duration = time.time() - speech_start
                    if duration >= MIN_SPEECH_DURATION:
                        print("   [Mic] Đã ngắt câu nói (hoàn thành thu âm).", flush=True)
                        result_frames.extend(speech_frames)
                        done.set()
                    else:
                        in_speech = False
                        speech_frames.clear()

        if in_speech and time.time() - speech_start > MAX_RECORD_DURATION:
            result_frames.extend(speech_frames)
            done.set()

    stream = sd.InputStream(
        samplerate = NATIVE_RATE,
        channels   = 1,
        device     = MIC_DEVICE,
        callback   = _callback,
        blocksize  = CHUNK_SAMPLES,
        dtype      = "float32",
    )

    t0 = time.time()
    with stream:
        while not done.is_set() and (time.time() - t0 < max_wait_sec):
            if stop_event and stop_event.is_set():
                break
            done.wait(timeout=0.1)

    if on_speech_end:
        on_speech_end()

    if not result_frames:
        return None

    audio = np.concatenate(result_frames, axis=0)
    if np.abs(audio).max() < 0.002:
        return None

    if NATIVE_RATE != WHISPER_RATE:
        audio = _resample(audio, NATIVE_RATE, WHISPER_RATE)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, WHISPER_RATE)
    tmp.close()
    return tmp.name


def record_audio() -> str | None:
    """Push-to-talk fallback (test thủ công)."""
    print("\n[Mic] Nhấn Enter để nói...", flush=True)
    input()
    frames = []

    def cb(indata, *_):
        frames.append(indata.copy())

    print("[Mic] Đang nghe... (Enter để dừng)", flush=True)
    with sd.InputStream(samplerate=NATIVE_RATE, channels=1,
                        device=MIC_DEVICE, callback=cb, dtype="float32"):
        input()

    if not frames:
        return None
    audio = np.concatenate(frames, axis=0)
    if NATIVE_RATE != WHISPER_RATE:
        audio = _resample(audio, NATIVE_RATE, WHISPER_RATE)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, WHISPER_RATE)
    tmp.close()
    return tmp.name
