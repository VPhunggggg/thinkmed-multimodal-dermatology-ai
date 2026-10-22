#!/usr/bin/env python3
"""
parakeet_onnx_run.py — Parakeet CTC ONNX + feature extraction KHỚP NeMo.

Feature extraction dùng ĐÚNG config NeMo:
  sample_rate=16000, window_size=0.025 (400 mẫu), window_stride=0.01 (160 mẫu),
  window=hann, n_fft=512, features=80 mel, log=True, normalize=per_feature,
  dither=1e-05.

Chạy:
    source ~/nemo_test_env/bin/activate
    pip install sentencepiece librosa
    ONNX_EP=cuda python parakeet_onnx_run.py <file.wav>
"""
import sys, os, time
import numpy as np

MODEL_DIR = os.path.expanduser("~/parakeet_onnx")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
SPM_PATH  = os.path.join(MODEL_DIR, "tokenizer.model")

# ── Tham số KHỚP NeMo config ──
SR          = 16000
N_FFT       = 512
WIN_LENGTH  = int(0.025 * SR)   # 400
HOP_LENGTH  = int(0.01 * SR)    # 160
N_MELS      = 80
DITHER      = 1e-05


def extract_features_nemo(y):
    """
    Mel-spectrogram KHỚP NeMo AudioToMelSpectrogramPreprocessor.
    Khác bản cũ: window=hann (đã đúng), THÊM dither, THÊM preemphasis KHÔNG có,
    log(x) với NeMo dùng log(x + 2^-24)... nhưng chuẩn là log sau mel.
    normalize=per_feature: chuẩn hóa TỪNG mel bin theo mean/std của nó.
    """
    import librosa

    # NeMo thêm dither (nhiễu nhỏ) trước khi tính — khớp lúc train
    if DITHER > 0:
        y = y + DITHER * np.random.randn(*y.shape).astype(y.dtype)

    # Mel-spectrogram: hann window, power=2 (magnitude^2)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        window="hann", n_mels=N_MELS, power=2.0, center=True,
        fmin=0.0, fmax=SR/2.0,
    )

    # log: NeMo dùng log(mel + 2^-24) — không phải 1e-9 như bản cũ
    logmel = np.log(mel + (2.0 ** -24))

    # normalize=per_feature: mỗi mel bin (mỗi hàng) chuẩn hóa riêng
    mean = logmel.mean(axis=1, keepdims=True)
    std  = logmel.std(axis=1, keepdims=True)
    logmel = (logmel - mean) / (std + 1e-5)

    return logmel.astype(np.float32)  # [80, T]


def ctc_decode(logprobs, blank_id):
    """CTC greedy: argmax → bỏ lặp liên tiếp → bỏ blank."""
    ids = logprobs.argmax(axis=-1)
    prev = -1
    tokens = []
    for i in ids:
        i = int(i)
        if i != prev and i != blank_id:
            tokens.append(i)
        prev = i
    return tokens


def main():
    if len(sys.argv) < 2:
        print("Dùng: python parakeet_onnx_run.py <file.wav>"); sys.exit(1)
    wav = sys.argv[1]
    ep = os.getenv("ONNX_EP", "cuda")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_PATH)
    vocab_size = sp.get_piece_size()

    import onnxruntime as ort
    ep_map = {
        "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu":      ["CPUExecutionProvider"],
    }
    sess = ort.InferenceSession(ONNX_PATH, providers=ep_map.get(ep, ep_map["cuda"]))

    import librosa
    y, _ = librosa.load(wav, sr=SR, mono=True)
    dur = len(y) / SR
    feat = extract_features_nemo(y)          # [80, T]
    feat_batch = feat[np.newaxis, :, :]      # [1, 80, T]
    length = np.array([feat.shape[1]], dtype=np.int64)

    inp = sess.get_inputs()
    feed = {inp[0].name: feat_batch}
    if len(inp) > 1:
        feed[inp[1].name] = length

    # Warmup + đo
    sess.run(None, feed)
    times = []
    for _ in range(3):
        t0 = time.time()
        outputs = sess.run(None, feed)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)

    logprobs = outputs[0]
    if logprobs.ndim == 3:
        logprobs = logprobs[0]               # [T, 1025]
    blank_id = logprobs.shape[-1] - 1        # 1024

    tokens = ctc_decode(logprobs, blank_id)
    text = sp.decode(tokens)

    print(f"\n{'='*60}")
    print(f"Provider: {ep}  |  Thời gian: {avg:.3f}s  |  RTF = {avg/dur:.3f}")
    print(f"Text: \"{text}\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
