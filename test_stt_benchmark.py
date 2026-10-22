#!/usr/bin/env python3
"""
stt_benchmark.py — Đo STT tách biệt, thử nhiều cấu hình để tìm cái nhanh nhất.

Chạy trên Jetson:
    cd ~/Desktop/skin_disease_rag
    python stt_benchmark.py path/to/test.wav

Nếu không có file wav sẵn, ghi âm 1 câu trước:
    python -c "from audio.recorder import record_audio; print(record_audio())"
"""
import os, sys, time

# Ép cấu hình TRƯỚC khi import (vì stt.py đọc env lúc import)
def bench(wav, device, beam, threads, compute):
    os.environ["STT_DEVICE"]      = device
    os.environ["STT_BEAM_SIZE"]   = str(beam)
    os.environ["STT_CPU_THREADS"] = str(threads)
    os.environ["STT_COMPUTE"]     = compute
    # reload module de doc lai env
    import importlib
    from faster_whisper import WhisperModel
    model_path = os.getenv("WHISPER_MODEL", "small")

    t0 = time.time()
    kw = dict(model_size_or_path=model_path, device=device, compute_type=compute)
    if device == "cpu":
        kw["cpu_threads"] = threads
    m = WhisperModel(**kw)
    t_load = time.time() - t0

    # Transcribe 2 lần, lấy lần 2 (lần 1 có warmup)
    for _ in range(2):
        t0 = time.time()
        segs, _ = m.transcribe(wav, language="vi", beam_size=beam,
                               condition_on_previous_text=False, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs)
        t_trans = time.time() - t0
    del m
    return t_load, t_trans, text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python stt_benchmark.py test.wav")
        sys.exit(1)
    wav = sys.argv[1]

    print(f"{'cấu hình':40s} {'load':>7s} {'trans':>7s}")
    print("-" * 60)
    configs = [
        ("cpu", 5, 1, "int8"),   # BẢN CŨ (baseline)
        ("cpu", 1, 1, "int8"),   # chỉ beam=1
        ("cpu", 5, 6, "int8"),   # chỉ threads=6
        ("cpu", 1, 6, "int8"),   # beam=1 + threads=6 (kỳ vọng nhanh nhất CPU)
    ]
    for dev, beam, thr, comp in configs:
        try:
            tl, tt, txt = bench(wav, dev, beam, thr, comp)
            print(f"{dev} beam={beam} threads={thr:<2d} {comp:5s}      "
                  f"{tl:6.1f}s {tt:6.2f}s   \"{txt[:30]}\"")
        except Exception as e:
            print(f"{dev} beam={beam} threads={thr} {comp}: LỖI {e}")
