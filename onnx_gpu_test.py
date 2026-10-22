#!/usr/bin/env python3
"""
onnx_gpu_test.py — Chạy Parakeet Vietnamese ONNX trên GPU (CUDA), đo tốc độ.

Môi trường đã xác nhận có CUDAExecutionProvider + TensorrtExecutionProvider.
Mục tiêu: đo RTF của Parakeet ONNX trên GPU, so với:
  - PhoWhisper CPU: RTF 0.52
  - Parakeet NeMo CPU: RTF 0.60

Chạy:
    source ~/nemo_test_env/bin/activate
    pip install onnx-asr
    python onnx_gpu_test.py /tmp/tmpwbfk1nkp.wav

Chọn provider qua env:
    ONNX_EP=cuda     (GPU CUDA — mặc định)
    ONNX_EP=tensorrt (TensorRT — nhanh nhất nhưng build lâu lần đầu)
    ONNX_EP=cpu      (CPU — để so sánh)
"""
import sys, os, time, wave


def audio_dur(p):
    with wave.open(p, 'rb') as w:
        return w.getnframes() / w.getframerate()


def main():
    if len(sys.argv) < 2:
        print("Dùng: python onnx_gpu_test.py <file.wav>"); sys.exit(1)
    wav = sys.argv[1]
    dur = audio_dur(wav)
    ep = os.getenv("ONNX_EP", "cuda")

    print(f"\n{'='*60}")
    print(f"File: {wav}  |  độ dài: {dur:.2f}s  |  provider: {ep}")
    print(f"{'='*60}\n")

    import onnxruntime as ort
    avail = ort.get_available_providers()
    print(f"[ORT] Providers khả dụng: {avail}")

    # Map lựa chọn -> tên provider onnxruntime
    ep_map = {
        "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu":      ["CPUExecutionProvider"],
    }
    providers = ep_map.get(ep, ep_map["cuda"])
    print(f"[ORT] Yêu cầu dùng: {providers[0]}\n")

    print("[1] Tải + load model Parakeet ONNX (lần đầu tải ~600MB)...")
    t0 = time.time()
    import onnx_asr
    # onnx-asr nhận providers để ép execution provider
    try:
        model = onnx_asr.load_model(
            "nemo-parakeet-ctc-0.6b",  # thử alias trước
            providers=providers,
        )
    except Exception:
        # dùng repo HuggingFace đã export
        model = onnx_asr.load_model(
            "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx",
            providers=providers,
        )
    print(f"    Load xong: {time.time()-t0:.1f}s")

    print("\n[2] Warmup (lần đầu, nếu TensorRT sẽ build engine — CÓ THỂ RẤT LÂU)...")
    t0 = time.time()
    model.recognize(wav)
    print(f"    Warmup: {time.time()-t0:.1f}s")

    print("\n[3] BATCH — đo 3 lần:")
    times = []
    for i in range(3):
        t0 = time.time()
        text = model.recognize(wav)
        dt = time.time() - t0
        times.append(dt)
        print(f"    Lần {i+1}: {dt:.2f}s")
    avg = sum(times) / len(times)
    rtf = avg / dur
    print(f"    → Trung bình: {avg:.2f}s  |  RTF = {rtf:.3f}")
    print(f'    → Text: "{text}"')

    print(f"\n{'='*60}")
    print("SO SÁNH RTF:")
    print(f"  PhoWhisper CPU:        0.52")
    print(f"  Parakeet NeMo CPU:     0.60")
    print(f"  Parakeet ONNX ({ep}):  {rtf:.3f}")
    if rtf < 0.30:
        print(f"  ✅✅ NHANH HƠN NHIỀU — GPU ăn tiền! Đây là lời giải.")
    elif rtf < 0.50:
        print(f"  ✅ Nhanh hơn — đáng dùng.")
    else:
        print(f"  ~ Chưa nhanh hơn CPU. Thử ONNX_EP=tensorrt, hoặc GPU chưa thật sự kích hoạt.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
