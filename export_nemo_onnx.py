#!/usr/bin/env python3
"""
export_nemo_onnx.py — Export model .nemo (Parakeet VN) sang ONNX KÈM tokenizer đúng.

Dùng file .nemo BẠN ĐÃ CÓ (chạy đúng qua NeMo) để export, đảm bảo tokenizer
tiếng Việt khớp — khác bản export sẵn của người khác bị sai tokenizer.

Chạy trong venv nemo:
    source ~/nemo_test_env/bin/activate
    python export_nemo_onnx.py

Xuất ra thư mục ~/parakeet_onnx/ gồm: model.onnx + tokenizer + config.
"""
import os, glob, shutil, json

# Tìm file .nemo đã tải
cands = glob.glob(os.path.expanduser("~/.cache/torch/NeMo/**/*.nemo"), recursive=True)
if not cands:
    raise SystemExit("Không tìm thấy .nemo — chạy lại nemo_streaming_test.py để tải model.")
nemo_path = cands[0]
print(f"File .nemo: {nemo_path}")

out_dir = os.path.expanduser("~/parakeet_onnx")
os.makedirs(out_dir, exist_ok=True)

print("Load model qua NeMo (CPU, tránh OOM)...")
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")

# 1. Export ONNX
onnx_path = os.path.join(out_dir, "model.onnx")
print(f"Export ONNX → {onnx_path} ...")
model.export(onnx_path)
print("   ✓ ONNX xong")

# 2. Lưu tokenizer/vocab (QUAN TRỌNG — đây là chỗ bản kia sai)
try:
    # SentencePiece model
    tok = model.tokenizer
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(tok.vocab_size):
            f.write(tok.ids_to_tokens([i])[0] + "\n")
    print(f"   ✓ Lưu vocab ({tok.vocab_size} token) → {vocab_path}")
except Exception as e:
    print(f"   ⚠ Không lưu được vocab tự động: {e}")
    print("   Thử tìm file .model SentencePiece trong .nemo...")

# 3. Giải nén .nemo tìm tokenizer.model (SentencePiece)
print("\nGiải nén .nemo tìm tokenizer SentencePiece...")
import tarfile
extract_dir = os.path.join(out_dir, "_nemo_extract")
os.makedirs(extract_dir, exist_ok=True)
try:
    with tarfile.open(nemo_path) as tar:
        tar.extractall(extract_dir)
    for spm in glob.glob(os.path.join(extract_dir, "**/*.model"), recursive=True):
        shutil.copy(spm, os.path.join(out_dir, "tokenizer.model"))
        print(f"   ✓ Copy tokenizer SentencePiece → tokenizer.model")
    for txt in glob.glob(os.path.join(extract_dir, "**/*vocab*"), recursive=True):
        shutil.copy(txt, out_dir)
        print(f"   ✓ Copy {os.path.basename(txt)}")
except Exception as e:
    print(f"   ⚠ {e}")

print(f"\n✅ Xong. Thư mục {out_dir} chứa:")
for f in sorted(os.listdir(out_dir)):
    if os.path.isfile(os.path.join(out_dir, f)):
        print(f"   {f}")
print("\nBước tiếp: dùng model.onnx + tokenizer.model này để chạy trên GPU với tokenizer ĐÚNG.")
