"""
test_voice.py — Kiểm tra từng phần của voice pipeline trước khi chạy toàn bộ.

Usage:
    python test_voice.py          # test tất cả
    python test_voice.py mic      # test mic
    python test_voice.py stt      # test STT
    python test_voice.py tts      # test TTS
    python test_voice.py full     # test end-to-end: nói → nhận ra → đọc lại
"""

import sys


def test_mic():
    print("\n=== TEST MIC ===")
    from audio.recorder import test_mic
    ok = test_mic(duration=2.0)
    return ok


def test_stt():
    print("\n=== TEST STT ===")
    from audio.stt import preload, transcribe
    from audio.recorder import record_audio

    preload()   # load model trước
    print("Sẵn sàng ghi âm. Thử nói: 'bệnh vảy nến'")
    wav = record_audio()
    if not wav:
        print("❌ Không ghi được âm thanh")
        return False

    text = transcribe(wav)
    if text:
        print(f"✅ Nhận ra: '{text}'")
        return True
    else:
        print("❌ Không nhận ra giọng nói")
        return False


def test_tts():
    print("\n=== TEST TTS ===")
    from audio.tts import test_tts
    ok = test_tts("Xin chào, tôi là trợ lý tư vấn bệnh da liễu.")
    if ok:
        print("✅ TTS hoạt động")
    else:
        print("❌ TTS lỗi — kiểm tra internet và ffmpeg")
    return ok


def test_full():
    print("\n=== TEST FULL PIPELINE ===")
    print("Nói một câu, hệ thống sẽ đọc lại để xác nhận nhận dạng đúng.\n")

    from audio.stt import preload, transcribe
    from audio.recorder import record_audio
    from audio.tts import speak

    preload()

    wav  = record_audio()
    if not wav:
        print("❌ Không ghi được âm thanh")
        return False

    text = transcribe(wav)
    if not text:
        print("❌ Không nhận ra giọng nói")
        return False

    print(f"📝 Nhận ra: '{text}'")
    print("🔊 Đọc lại...")
    speak(f"Bạn vừa nói: {text}")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}

    if cmd in ("mic", "all"):
        results["mic"] = test_mic()

    if cmd in ("tts", "all"):
        results["tts"] = test_tts()

    if cmd in ("stt", "all"):
        results["stt"] = test_stt()

    if cmd in ("full",):
        results["full"] = test_full()

    if results:
        print("\n=== KẾT QUẢ ===")
        for name, ok in results.items():
            icon = "✅" if ok else "❌"
            print(f"  {icon} {name.upper()}")

        if all(results.values()):
            print("\n🎉 Tất cả OK! Chạy: python main.py --image ảnh.jpg --voice")
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"\n⚠️  Cần fix: {', '.join(failed)}")
