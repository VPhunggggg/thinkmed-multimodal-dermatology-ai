#!/usr/bin/env python3
"""Chẩn đoán model ONNX + tokenizer tải về."""
import os, glob

print("="*60)
print("1. FILE MODEL ONNX TẢI VỀ")
print("="*60)
# onnx-asr cache thường ở ~/.cache/huggingface
patterns = [
    os.path.expanduser("~/.cache/huggingface/**/*parakeet*vietnamese*"),
    os.path.expanduser("~/.cache/**/*parakeet*vietnamese*onnx*"),
]
found_dir = None
for pat in patterns:
    for p in glob.glob(pat, recursive=True):
        if os.path.isdir(p):
            found_dir = p
            print(f"\nThư mục: {p}")
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                sz = os.path.getsize(fp)/(1024*1024) if os.path.isfile(fp) else 0
                print(f"   {f}  ({sz:.1f} MB)" if sz else f"   {f}/")

# Tìm mọi file liên quan tokenizer/vocab
print("\n" + "="*60)
print("2. FILE TOKENIZER / VOCAB")
print("="*60)
for kw in ["vocab", "token", "*.model", "*.txt", "*.json"]:
    for pat in [os.path.expanduser(f"~/.cache/huggingface/**/*parakeet*vietnamese*/**/{kw}"),
                os.path.expanduser(f"~/.cache/huggingface/**/*parakeet*vietnamese*/{kw}")]:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p):
                print(f"   {p} ({os.path.getsize(p)} bytes)")

print("\n" + "="*60)
print("3. onnx-asr — model hỗ trợ")
print("="*60)
try:
    import onnx_asr
    print(f"   onnx_asr version: {onnx_asr.__version__ if hasattr(onnx_asr,'__version__') else '?'}")
    # xem danh sách model onnx-asr biết
    if hasattr(onnx_asr, "_models") or hasattr(onnx_asr, "list_models"):
        print("   (có hàm liệt kê model)")
except Exception as e:
    print(f"   {e}")
