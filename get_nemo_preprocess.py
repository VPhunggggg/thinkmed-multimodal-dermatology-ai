#!/usr/bin/env python3
"""
get_nemo_preprocess.py — Đọc CHÍNH XÁC tham số preprocessing của model NeMo.

Feature extraction phải khớp NeMo 100% thì decode mới hết lỗi dấu.
Script này in ra config preprocessor để dùng đúng tham số.
"""
import glob, os, json

cands = glob.glob(os.path.expanduser("~/.cache/torch/NeMo/**/*.nemo"), recursive=True)
nemo_path = cands[0]
print(f"File .nemo: {nemo_path}\n")

# Cách 1: load qua NeMo, đọc cfg.preprocessor
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")

print("="*60)
print("CONFIG PREPROCESSOR (dùng đúng các số này)")
print("="*60)
prep = model.cfg.preprocessor
for k, v in prep.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("CONFIG DECODER (xem blank_id, vocab)")
print("="*60)
try:
    dec = model.cfg.decoder
    for k in ["num_classes", "vocabulary"]:
        if k in dec:
            val = dec[k]
            if k == "vocabulary" and isinstance(val, list):
                print(f"  {k}: [{len(val)} phần tử] {val[:5]}...")
            else:
                print(f"  {k}: {val}")
except Exception as e:
    print(f"  {e}")

# Lưu ra file JSON để script chạy đọc
out = {}
for k, v in prep.items():
    try:
        json.dumps(v)
        out[k] = v
    except Exception:
        out[k] = str(v)
with open(os.path.expanduser("~/parakeet_onnx/preprocess_config.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n✓ Đã lưu config → ~/parakeet_onnx/preprocess_config.json")
