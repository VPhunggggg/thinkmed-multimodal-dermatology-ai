#!/usr/bin/env python3
"""
nemo_streaming_test.py — Test Parakeet Vietnamese (FastConformer) so với PhoWhisper.

Chạy TRONG môi trường NeMo tách bạch:
    source ~/nemo_test_env/bin/activate
    python nemo_streaming_test.py /tmp/test.wav

Đo:
  1. Batch RTF của Parakeet (so với PhoWhisper RTF 0.52 đã đo)
  2. Streaming thật (nếu dùng cache-aware streaming model)
  3. Độ chính xác — QUAN TRỌNG: nghe câu "kết thúc hội thoại" đúng không

MỤC TIÊU: chứng minh Parakeet streaming NHANH HƠN PhoWhisper batch,
TRƯỚC khi tích hợp vào ThinkMed.
"""
import sys, time, wave
import numpy as np


def audio_dur(p):
    with wave.open(p,'rb') as w: return w.getnframes()/w.getframerate()


def main():
    if len(sys.argv) < 2:
        print("Dùng: python nemo_streaming_test.py <file.wav>"); sys.exit(1)
    wav = sys.argv[1]
    dur = audio_dur(wav)

    print(f"\n{'='*60}\nFile: {wav}  |  độ dài: {dur:.2f}s\n{'='*60}")

    print("\n[1] Tải model Parakeet Vietnamese...")
    t0 = time.time()
    import nemo.collections.asr as nemo_asr
    import glob, os
    import torch
    # QUAN TRỌNG: load lên CPU trước (map_location='cpu') để tránh OOM VRAM lúc
    # load. Orin Nano 8GB chia sẻ — model 0.6B + NeMo runtime dễ vượt VRAM nếu
    # load thẳng lên GPU. Sau khi load xong sẽ thử chuyển GPU CÓ KIỂM SOÁT.
    STT_DEV = os.getenv("PARAKEET_DEVICE", "cpu")   # 'cpu' an toàn, 'cuda' nếu đủ VRAM

    def find_nemo():
        c = glob.glob(os.path.expanduser("~/.cache/torch/NeMo/**/*.nemo"), recursive=True)
        c += glob.glob(os.path.expanduser("~/.cache/huggingface/**/*.nemo"), recursive=True)
        return c

    cands = find_nemo()
    if cands:
        print(f"    Load từ .nemo: {cands[0]}")
        model = nemo_asr.models.ASRModel.restore_from(cands[0], map_location=torch.device("cpu"))
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-ctc-0.6b-Vietnamese", map_location=torch.device("cpu"))
    print(f"    Load xong (CPU): {time.time()-t0:.1f}s")

    dev = STT_DEV
    print(f"    Thử chuyển sang device: {dev}")
    if dev == "cuda":
        try:
            model = model.cuda()
            print("    ✅ Đã lên GPU")
        except Exception as e:
            print(f"    ⚠ Không lên được GPU ({e}) — chạy CPU")
            dev = "cpu"
    else:
        print("    (Chạy trên CPU — an toàn cho VRAM. Đặt PARAKEET_DEVICE=cuda để thử GPU)")

    # Warmup
    print("\n[2] Warmup...")
    model.transcribe([wav])

    # Batch
    print("\n[3] BATCH — đo 3 lần:")
    times = []
    for i in range(3):
        t0 = time.time()
        out = model.transcribe([wav])
        dt = time.time() - t0
        times.append(dt)
        print(f"    Lần {i+1}: {dt:.2f}s")
    avg = sum(times)/len(times)
    rtf = avg / dur
    # NeMo mới trả list Hypothesis object -> lấy .text; bản cũ trả list str
    r0 = out[0] if isinstance(out, (list, tuple)) else out
    text = getattr(r0, "text", r0)  # Hypothesis.text hoặc chuỗi thẳng
    print(f"    → Trung bình: {avg:.2f}s  |  RTF = {rtf:.3f}")
    print(f'    → Text: "{text}"')

    print(f"\n{'='*60}")
    print("SO SÁNH:")
    print(f"  PhoWhisper-small (đã đo): RTF ~0.52, batch ~6.8s cho câu 13s")
    print(f"  Parakeet Vietnamese:      RTF {rtf:.3f}, batch {avg:.2f}s")
    if rtf < 0.52:
        print(f"  ✅ Parakeet NHANH HƠN PhoWhisper {0.52/rtf:.1f}x")
    else:
        print(f"  ⚠  Parakeet không nhanh hơn — cân nhắc lại")
    print(f"\n  Kiểm tra độ chính xác thủ công: text trên có đúng câu bạn nói?")
    print(f"  Đặc biệt thử câu 'kết thúc hội thoại' (PhoWhisper chỉ đúng 30%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
