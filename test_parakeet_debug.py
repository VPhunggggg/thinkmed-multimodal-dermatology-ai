#!/usr/bin/env python3
"""
test_parakeet_debug.py — Test stt_parakeet với chẩn đoán từng bước, KHÔNG xóa file.

Chạy trong venv:
    source ~/nemo_test_env/bin/activate
    # Ghi âm mới:
    python -c "from audio.recorder import record_audio; import shutil; p=record_audio(); shutil.copy(p,'/tmp/keep.wav'); print('Đã lưu /tmp/keep.wav')"
    python test_parakeet_debug.py /tmp/keep.wav
"""
import sys, os, time
import numpy as np

if len(sys.argv) < 2:
    print("Dùng: python test_parakeet_debug.py <file.wav>"); sys.exit(1)
wav = sys.argv[1]

print(f"File: {wav}")
print(f"Tồn tại: {os.path.exists(wav)}")
if not os.path.exists(wav):
    print("❌ File không tồn tại! (Có thể đã bị xóa lần chạy trước)")
    sys.exit(1)
print(f"Kích thước: {os.path.getsize(wav)} bytes\n")

# Đọc audio, xem độ dài
import librosa
y, sr = librosa.load(wav, sr=16000, mono=True)
dur = len(y) / 16000
print(f"Audio: {dur:.2f}s ({len(y)} mẫu @ 16kHz)")
print(f"Biên độ: max={np.abs(y).max():.4f}, mean={np.abs(y).mean():.4f}")
if dur < 0.2:
    print("❌ Audio < 0.2s → module coi là im lặng, trả None")
    sys.exit(1)
if np.abs(y).max() < 0.001:
    print("❌ Audio gần như im lặng (biên độ ~0) → có thể mic không thu được")
    sys.exit(1)
print("✓ Audio hợp lệ\n")

# Gọi thẳng các bước của module
sys.path.insert(0, "audio")
import stt_parakeet as P

print("Load session...")
sess, sp = P._get_session()

print("\nTrích features...")
feat = P._extract_features(y)
print(f"  Features shape: {feat.shape}")
print(f"  Features range: [{feat.min():.2f}, {feat.max():.2f}]")

print("\nChạy ONNX...")
feed = {sess.get_inputs()[0].name: feat[np.newaxis, :, :]}
if len(sess.get_inputs()) > 1:
    feed[sess.get_inputs()[1].name] = np.array([feat.shape[1]], dtype=np.int64)
t0 = time.time()
outputs = sess.run(None, feed)
dt = time.time() - t0
logprobs = outputs[0]
if logprobs.ndim == 3:
    logprobs = logprobs[0]
print(f"  Logprobs shape: {logprobs.shape}  |  {dt*1000:.0f}ms  |  RTF={dt/dur:.3f}")

blank_id = logprobs.shape[-1] - 1
tokens = P._ctc_decode(logprobs, blank_id)
print(f"  Số token decode: {len(tokens)}")
print(f"  Token đầu: {tokens[:20]}")

text = sp.decode(tokens).strip()
print(f"\n{'='*60}")
print(f"TEXT: \"{text}\"")
print(f"{'='*60}")
if not text:
    print("❌ Text rỗng — có thể toàn blank token, hoặc feature sai")
