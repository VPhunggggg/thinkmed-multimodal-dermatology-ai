"""
test_voice_fuzzy.py — Test find_disease_fuzzy() với giọng nói THẬT qua mic,
không phải text gõ tay giả lập lỗi.

Khác với fuzzy_match.py (chạy trực tiếp, dùng string mẫu), script này:
  mic → VAD → faster-whisper STT → find_disease_fuzzy()
và hiện rõ text STT thực sự nhận được (có thể sai khác test mẫu),
cùng top 3 candidate + điểm số để biết ngưỡng 80% có phù hợp không.

Usage:
    python test_voice_fuzzy.py
    (Ctrl+C hoặc nói "thoát" để dừng và xem tổng kết)
"""

import sys
import time

from audio.recorder import record_vad
from audio.stt import preload, transcribe
from fuzzy_match import find_disease_fuzzy, debug_match

EXIT_WORDS = {"thoát", "thoat", "exit", "q"}


def main():
    print("=" * 60)
    print("TEST FUZZY MATCHING VỚI GIỌNG NÓI THẬT")
    print("=" * 60)
    print("Preloading STT model...")
    preload()
    print("\nSẵn sàng. Hãy thử nói các câu như:")
    print('  - "tôi bị vảy nến"')
    print('  - "trị bệnh vảy nến như thế nào" (cố tình nói nhanh/không rõ)')
    print('  - "bệnh phong có nguy hiểm không"')
    print("Nói 'thoát' để dừng.\n")

    results = []   # (stt_text, matched_key, top_score, top_alias)

    round_n = 0
    try:
        while True:
            round_n += 1
            print(f"\n{'─' * 50}")
            print(f"Lượt {round_n} — nói ngay bây giờ...")

            wav_path = record_vad(max_wait_sec=15.0)

            if wav_path is None:
                print("⚠️  Không nghe được gì (im lặng/timeout). Thử lại.")
                continue

            text = transcribe(wav_path)

            if not text:
                print("⚠️  STT không nhận ra được nội dung.")
                continue

            print(f"📝 STT nhận được: \"{text}\"")

            if text.lower().strip() in EXIT_WORDS:
                print("→ Nhận diện lệnh thoát.")
                break

            debug = debug_match(text)
            matched = debug["result"]

            print(f"🔎 Chuẩn hoá:      \"{debug['normalized']}\"")
            print(f"🏆 Top 3 candidate:")
            for key, alias, score in debug["top_matches"][:3]:
                marker = "  ✅" if key == matched and score >= 80 else "    "
                print(f"{marker} {key:20} ~ '{alias}'  → {score:.0f}%")

            if matched:
                print(f"➡️  KẾT QUẢ: nhận diện bệnh = {matched}")
            else:
                print(f"➡️  KẾT QUẢ: KHÔNG nhận diện được bệnh nào (dưới ngưỡng 80%)")

            top_key, top_alias, top_score = debug["top_matches"][0]
            results.append((text, matched, top_score, top_alias))

    except KeyboardInterrupt:
        print("\n\n(Dừng bởi Ctrl+C)")

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TỔNG KẾT — {len(results)} lượt test")
    print("=" * 60)

    if not results:
        print("Chưa có dữ liệu nào để tổng kết.")
        return

    matched_count = sum(1 for _, m, _, _ in results if m)
    print(f"Nhận diện được bệnh: {matched_count}/{len(results)}")
    print()
    print(f"{'STT text':<35} {'Matched':<15} {'Score':<8} Closest alias")
    print("-" * 80)
    for text, matched, score, alias in results:
        m_str = matched or "—"
        print(f"{text[:33]:<35} {m_str:<15} {score:<7.0f} {alias}")

    print()
    low_confidence = [r for r in results if 60 <= r[2] < 80]
    if low_confidence:
        print(f"⚠️  {len(low_confidence)} câu có điểm 60-79% (gần ngưỡng, có thể cần hạ FUZZY_THRESHOLD):")
        for text, matched, score, alias in low_confidence:
            print(f"   '{text}' → {score:.0f}% với '{alias}'")


if __name__ == "__main__":
    main()
