#!/usr/bin/env python3
"""Xem vocab.txt của bản ONNX có phải tiếng Việt không."""
import os

vp = os.path.expanduser("~/.cache/huggingface/hub/models--OpenVoiceOS--nvidia-parakeet-ctc-0.6b-vietnamese-onnx/snapshots/ed9f55ba980eb1c9eeba02a5733eba7cba02f6e7/vocab.txt")

with open(vp, encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]

print(f"Tổng số token: {len(lines)}")
print(f"\n30 token đầu:")
for i, t in enumerate(lines[:30]):
    print(f"  [{i}] {repr(t)}")

print(f"\n30 token giữa:")
mid = len(lines)//2
for i, t in enumerate(lines[mid:mid+30], mid):
    print(f"  [{i}] {repr(t)}")

# Đếm token có dấu tiếng Việt
vn_chars = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
has_vn = sum(1 for t in lines if any(c in vn_chars for c in t.lower()))
print(f"\nSố token CÓ dấu tiếng Việt: {has_vn}/{len(lines)}")
if has_vn < 20:
    print("❌ Vocab KHÔNG phải tiếng Việt — đây là lý do text ra rác!")
else:
    print("✓ Vocab có tiếng Việt")

# Xem config.json
cp = os.path.expanduser("~/.cache/huggingface/hub/models--OpenVoiceOS--nvidia-parakeet-ctc-0.6b-vietnamese-onnx/snapshots/ed9f55ba980eb1c9eeba02a5733eba7cba02f6e7/config.json")
print(f"\nconfig.json:")
print(open(cp).read())

