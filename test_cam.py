"""
test_camera.py — Test camera riêng biệt, KHÔNG load RAG/YOLO/UI.

So sánh 2 cách đọc camera:
  A) cv2.VideoCapture(0)              — cách robot_main.py hiện đang dùng
  B) nvarguscamerasrc qua GStreamer   — cách chuẩn cho CSI camera trên Jetson

Lưu ảnh ra file + in thống kê màu RGB để chẩn đoán nhanh.
"""

import time
import cv2
import numpy as np


def report(frame, label, out_dir="/home/fptuadmin"):
    if frame is None:
        print(f"[{label}] ❌ Không đọc được frame")
        return False

    h, w = frame.shape[:2]
    b = float(frame[:, :, 0].mean())
    g = float(frame[:, :, 1].mean())
    r = float(frame[:, :, 2].mean())
    print(f"[{label}] ✅ Frame {w}x{h} — B={b:.0f} G={g:.0f} R={r:.0f}")

    if g > r * 1.8 and g > b * 1.8:
        print(f"[{label}] ⚠️  Kênh xanh lá cao bất thường — đúng là raw Bayer chưa decode")

    out_path = f"{out_dir}/test_camera_{label}.jpg"
    cv2.imwrite(out_path, frame)
    print(f"[{label}] Đã lưu: {out_path}")
    return True


print("=" * 55)
print("TEST A: cv2.VideoCapture(0) — cách robot_main.py đang dùng")
print("=" * 55)
cap = cv2.VideoCapture(0)
time.sleep(1)
ok, frame = cap.read()
report(frame if ok else None, "A_v4l2")
cap.release()

print()
print("=" * 55)
print("TEST B: nvarguscamerasrc (GStreamer CSI pipeline chuẩn Jetson)")
print("=" * 55)
gst = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap2 = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
if not cap2.isOpened():
    print("[B_nvargus] ❌ Không mở được pipeline GStreamer")
else:
    time.sleep(1)
    ok2, frame2 = cap2.read()
    report(frame2 if ok2 else None, "B_nvargus")
cap2.release()

print("\nMở 2 file .jpg để so sánh — file nào màu đúng (không xanh lá) là cách cần dùng.")
