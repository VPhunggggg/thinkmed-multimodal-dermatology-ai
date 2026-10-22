"""
test_voice_io.py — GIAI ĐOẠN 4c: kiểm tra giọng nói (mic + loa) độc lập.

KHÔNG camera, KHÔNG YOLO, KHÔNG UI. Chỉ test 3 mảnh:
  A. TTS  → loa      (edge-tts → loa Bluetooth device 33)
  B. STT  → chữ      (mic device 0 → faster-whisper)
  C. Vòng lặp đầy đủ (nói → nhận chữ → đọc lại)

CẦN: ngồi cạnh mic và loa để nói + nghe.
Chạy:
    cd ~/Desktop/skin_disease_rag
    python3 test_voice_io.py
"""

import sys

# ── Kiểm tra thư viện, báo rõ nếu thiếu ───────────────────────────────────────
try:
    from audio.tts import speak
    from audio.stt import transcribe, preload
    from audio.recorder import record_vad, record_audio, MIC_DEVICE
except ImportError as e:
    sys.exit(
        f"❌ Thiếu thư viện: {e}\n"
        "   Cài:  pip install sounddevice soundfile soxr faster-whisper edge-tts\n"
        "         sudo apt install -y ffmpeg"
    )

print("=" * 70)
print(f"[Voice] Mic device = {MIC_DEVICE} | TTS qua loa mặc định (PulseAudio)")
print("=" * 70)


# ── PHẦN A: TTS → loa ─────────────────────────────────────────────────────────
print("\n▶ PHẦN A — Test loa (TTS)")
print("  Sắp đọc 1 câu qua loa. Chú ý nghe...")
input("  Nhấn Enter để phát: ")

ok = speak("Xin chào. Đây là bài kiểm tra loa của hệ thống tư vấn da liễu.")
if not ok:
    print("  ⚠️  speak() trả về False — kiểm tra: loa Bluetooth đã kết nối? "
          "player (ffplay) đã cài? có mạng cho edge-tts?")

heard = input("  Bạn có NGHE thấy câu vừa rồi không? [y/n]: ").strip().lower()
tts_ok = heard.startswith("y")
print("  → TTS:", "✅ OK" if tts_ok else "❌ chưa nghe được")


# ── PHẦN B: mic → STT ─────────────────────────────────────────────────────────
print("\n▶ PHẦN B — Test mic + nhận dạng giọng nói (STT)")
print("  Đang nạp model whisper (lần đầu có thể tải ~500MB, chờ chút)...")
preload()
print("  ✅ Model sẵn sàng.")

SUGGEST = "tôi bị vảy nến ở khuỷu tay"
print(f'\n  Gợi ý: hãy nói rõ câu — "{SUGGEST}" — để dễ đối chiếu độ chính xác.')
print("  Chế độ VAD: cứ nói, hệ thống tự nhận biết bắt đầu/kết thúc.")

text = record_vad(
    on_listening_start = lambda: print("  🎤 Đang nghe... (nói đi)"),
    on_speech_start    = lambda: print("  🔴 Phát hiện giọng nói..."),
    on_speech_end      = lambda: print("  ⏹  Kết thúc, đang xử lý..."),
    max_wait_sec       = 20.0,
)

# Fallback push-to-talk nếu VAD không bắt được (ngưỡng im lặng chưa hợp mic)
if text is None:
    print("  ⚠️  VAD không bắt được giọng (im lặng/ngưỡng chưa hợp).")
    print("      Thử lại bằng chế độ nhấn-Enter:")
    wav = record_audio()          # Enter để bắt đầu, Enter để dừng
    text = transcribe(wav) if wav else None
else:
    text = transcribe(text)

print(f"\n  📝 Nhận dạng được: {text!r}")
stt_ok = bool(text and text.strip())
print("  → STT:", "✅ có ra chữ" if stt_ok else "❌ không ra chữ")


# ── PHẦN C: vòng lặp đầy đủ ───────────────────────────────────────────────────
if tts_ok and stt_ok:
    print("\n▶ PHẦN C — Vòng lặp đầy đủ: đọc lại câu bạn vừa nói")
    speak(f"Bạn vừa nói: {text}")
    print("  → Nếu bạn nghe loa đọc lại đúng câu → mic + STT + TTS thông suốt.")


# ── TỔNG KẾT ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  TTS (loa): {'✅' if tts_ok else '❌'}   |   STT (mic): {'✅' if stt_ok else '❌'}")
if tts_ok and stt_ok:
    print("✅ GIAI ĐOẠN 4c PASS: giọng nói vào–ra hoạt động độc lập.")
else:
    print("⚠️  Còn mảnh chưa thông — xem cảnh báo ở phần tương ứng phía trên.")
