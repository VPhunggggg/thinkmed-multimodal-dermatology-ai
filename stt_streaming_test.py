#!/usr/bin/env python3
"""
stt_streaming_test.py — Đo khả năng streaming STT (nhận diện song song lúc nói).

Đo HAI thứ:
  1. RTF (real-time factor) = thời_gian_xử_lý / độ_dài_audio
     - RTF < 1  → model NHANH HƠN real-time → streaming KHẢ THI
     - RTF > 1  → model chậm hơn real-time → streaming KHÔNG cứu được
  2. Mô phỏng streaming: chia audio thành chunk, xử lý từng chunk như thể
     đang thu, đo "độ trễ cảm nhận" = thời gian từ lúc NGƯỜI NGỪNG NÓI
     đến khi có kết quả cuối (chỉ còn chunk cuối cần xử lý).

So sánh với chế độ batch (xử lý cả câu 1 lần) để thấy streaming lợi bao nhiêu.

Chạy TRÊN JETSON:
    cd ~/Desktop/skin_disease_rag
    # Ghi âm 1 câu test trước (nói ~4-5 giây, câu hỏi da liễu):
    python -c "from audio.recorder import record_audio; print(record_audio())"
    # Rồi:
    python stt_streaming_test.py /tmp/<file>.wav

LƯU Ý CÔNG SUẤT: kết quả phụ thuộc mạnh vào chế độ nguồn Jetson.
    sudo nvpmodel -q          # xem chế độ hiện tại
    sudo nvpmodel -m 0        # MAXN (hiệu năng tối đa) — nên dùng để test
    sudo jetson_clocks        # khóa xung nhịp ở mức cao nhất
"""
import os, sys, time, wave
import numpy as np


def audio_duration(wav_path):
    with wave.open(wav_path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def load_audio(wav_path, target_sr=16000):
    """Đọc WAV thành float32 mono 16kHz."""
    with wave.open(wav_path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        ch = w.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    # faster-whisper tự resample, nhưng ta giả định file đã 16kHz từ recorder
    return a, sr


def make_model():
    from faster_whisper import WhisperModel
    model_path = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("STT_DEVICE", "cpu")
    compute = os.getenv("STT_COMPUTE", "int8")
    threads = int(os.getenv("STT_CPU_THREADS", "3"))
    beam = int(os.getenv("STT_BEAM_SIZE", "3"))
    print(f"[Model] {model_path} | {device}/{compute} | threads={threads} beam={beam}")
    kw = dict(model_size_or_path=model_path, device=device, compute_type=compute)
    if device == "cpu":
        kw["cpu_threads"] = threads
    m = WhisperModel(**kw)
    return m, beam


def transcribe_array(model, audio, beam):
    """Transcribe một mảng numpy, trả (text, thời_gian)."""
    t0 = time.time()
    segs, _ = model.transcribe(audio, language="vi", beam_size=beam,
                               condition_on_previous_text=False, vad_filter=False)
    text = " ".join(s.text.strip() for s in segs)
    return text, time.time() - t0


def main():
    if len(sys.argv) < 2:
        print("Dùng: python stt_streaming_test.py <file.wav>")
        sys.exit(1)
    wav = sys.argv[1]
    if not os.path.exists(wav):
        print(f"Không tìm thấy file: {wav}"); sys.exit(1)

    dur = audio_duration(wav)
    audio, sr = load_audio(wav)
    print(f"\n{'='*64}")
    print(f"File: {wav}")
    print(f"Độ dài audio: {dur:.2f}s  ({len(audio)} mẫu @ {sr}Hz)")
    print(f"{'='*64}\n")

    model, beam = make_model()
    # Warmup (lần đầu chậm do khởi tạo)
    print("[Warmup] chạy 1 lần bỏ đi...")
    transcribe_array(model, audio[:sr], beam)

    # ── 1. BATCH: xử lý cả câu 1 lần (chế độ hiện tại) ──
    print(f"\n{'─'*64}")
    print("1. BATCH (xử lý cả câu — chế độ hiện tại của robot)")
    print(f"{'─'*64}")
    times = []
    for i in range(3):
        text, dt = transcribe_array(model, audio, beam)
        times.append(dt)
        print(f"  Lần {i+1}: {dt:.2f}s")
    batch_avg = sum(times) / len(times)
    rtf = batch_avg / dur
    print(f"  → Trung bình: {batch_avg:.2f}s")
    print(f"  → RTF (real-time factor) = {rtf:.2f}")
    if rtf < 1:
        print(f"  ✅ RTF < 1 → model nhanh hơn real-time → STREAMING KHẢ THI")
    else:
        print(f"  ❌ RTF > 1 → model CHẬM hơn real-time {rtf:.1f}x → streaming KHÓ cứu")
    print(f'  Text: "{text}"')

    # ── 2. STREAMING mô phỏng: chia chunk, xử lý lũy tiến ──
    # Mô phỏng: thu xong chunk nào xử lý chunk đó. "Độ trễ cảm nhận" =
    # thời gian xử lý CHUNK CUỐI (các chunk trước đã xử lý xong trong lúc nói,
    # NẾU model theo kịp). Nếu model không kịp, phần dồn ứ cộng vào cuối.
    print(f"\n{'─'*64}")
    print("2. STREAMING mô phỏng (chia chunk, xử lý song song lúc nói)")
    print(f"{'─'*64}")
    for chunk_sec in [2.0, 3.0]:
        chunk_n = int(chunk_sec * sr)
        chunks = [audio[i:i+chunk_n] for i in range(0, len(audio), chunk_n)]
        print(f"\n  Chunk = {chunk_sec}s  ({len(chunks)} đoạn):")

        # Mô phỏng thời gian: mỗi chunk "đến" sau mỗi chunk_sec giây (lúc nói).
        # Model xử lý tuần tự. Nếu xử lý chunk chậm hơn chunk_sec, hàng đợi dồn.
        arrival = 0.0          # thời điểm chunk đến (giây, tính từ đầu câu)
        proc_free = 0.0        # thời điểm model rảnh để nhận chunk mới
        last_done = 0.0        # thời điểm chunk cuối xử lý xong
        total_proc = 0.0
        for idx, c in enumerate(chunks):
            arrival = (idx + 1) * chunk_sec   # chunk idx đến khi đã nói xong nó
            # cumulative transcribe (giống streaming thật: xử lý audio tới thời điểm này)
            cum_audio = audio[:(idx+1)*chunk_n]
            _, dt = transcribe_array(model, cum_audio, beam)
            total_proc += dt
            start = max(arrival, proc_free)
            done = start + dt
            proc_free = done
            last_done = done
        speech_end = len(chunks) * chunk_sec
        perceived = max(0.0, last_done - speech_end)
        print(f"    Người ngừng nói tại: {speech_end:.2f}s")
        print(f"    Kết quả cuối xong tại: {last_done:.2f}s")
        print(f"    → Độ trễ CẢM NHẬN (sau khi ngừng nói): {perceived:.2f}s")
        print(f"    (tổng thời gian model xử lý: {total_proc:.2f}s)")

    print(f"\n{'='*64}")
    print("KẾT LUẬN:")
    print(f"  - Batch hiện tại: chờ {batch_avg:.1f}s sau khi ngừng nói")
    print(f"  - RTF = {rtf:.2f} ({'streaming khả thi' if rtf<1 else 'streaming KHÓ ăn vì model chậm hơn real-time'})")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
