#!/usr/bin/env python3
"""
onnx_asr_test.py — Test Parakeet Vietnamese ONNX qua onnx-asr, GPU vs CPU.

Model đã export sẵn: OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx
(chính là nvidia/parakeet-ctc-0.6b-Vietnamese, convert sang ONNX, có fp32 + int8)

Mục tiêu (A): chạy Parakeet trên GPU qua ONNX Runtime — thứ NeMo không làm được.
So RTF với: PhoWhisper CPU 0.52, Parakeet NeMo CPU 0.60.

Chạy trong venv (có thể dùng lại nemo_test_env hoặc tạo mới):
    pip install onnx-asr onnxruntime-gpu
    python onnx_asr_test.py /tmp/tmpwbfk1nkp.wav

Chọn provider qua env:
    ONNX_PROVIDER=cuda  (thử GPU)
    ONNX_PROVIDER=cpu   (CPU, mặc định để an toàn)
"""
import sys, os, time, wave


def audio_dur(p):
    with wave.open(p, 'rb') as w:
        return w.getnframes() / w.getframerate()


def main():
    if len(sys.argv) < 2:
        print("Dùng: python onnx_asr_test.py <file.wav>"); sys.exit(1)
    wav = sys.argv[1]
    dur = audio_dur(wav)
    provider = os.getenv("ONNX_PROVIDER", "cpu")

    print(f"\n{'='*60}")
    print(f"File: {wav}  |  độ dài: {dur:.2f}s")
    print(f"Provider yêu cầu: {provider}")
    print(f"{'='*60}\n")

    # Xem ONNX Runtime thấy provider nào
    import onnxruntime as ort
    avail = ort.get_available_providers()
    print(f"[ORT] Providers khả dụng: {avail}")
    if provider == "cuda" and "CUDAExecutionProvider" not in avail:
        print("[ORT] ⚠  CUDAExecutionProvider KHÔNG có — cần onnxruntime-gpu.")
        print("     Chạy CPU thay thế. Cài GPU: pip install onnxruntime-gpu")

    print("\n[1] Tải + load model ONNX (lần đầu tải ~600MB)...")
    t0 = time.time()
    import onnx_asr
    # onnx-asr chọn provider theo onnxruntime cài đặt; ép CUDA nếu có
    providers = None
    if provider == "cuda" and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        if providers:
            model = onnx_asr.load_model(
                "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx",
                providers=providers)
        else:
            model = onnx_asr.load_model(
                "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx")
    except TypeError:
        # bản onnx-asr cũ không nhận 'providers' -> load thường
        model = onnx_asr.load_model(
            "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx")
    print(f"    Load xong: {time.time()-t0:.1f}s")

    print("\n[2] Warmup...")
    model.recognize(wav)

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
    print(f"  PhoWhisper CPU:      0.52")
    print(f"  Parakeet NeMo CPU:   0.60")
    print(f"  Parakeet ONNX ({provider}):  {rtf:.3f}")
    if rtf < 0.45:
        print(f"  ✅ NHANH HƠN đáng kể — GPU (hoặc ONNX tối ưu) ăn tiền")
    elif rtf < 0.65:
        print(f"  ~ Ngang CPU — GPU chưa kích hoạt hoặc không giúp")
    print(f"\n  Kiểm tra text có đúng không (đặc biệt câu khó như 'kết thúc hội thoại')")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
