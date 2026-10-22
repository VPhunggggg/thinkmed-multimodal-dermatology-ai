"""
test_yolo_latency.py — GIAI ĐOẠN 4a: đo tốc độ YOLO trên ảnh tĩnh.

Chạy YOLO ĐÚNG cách robot thật chạy (_run_yolo trong robot_main.py):
    model(frame_numpy, conf=DETECT_CONF_MIN)  → tự dùng GPU nếu có.

Mục đích:
  1. Xác nhận model detect đúng bệnh trên ảnh mẫu.
  2. Đo ms/frame trên GPU — CON SỐ QUYẾT ĐỊNH: YOLO có đủ nhanh cho kiosk
     không, hay cần tối ưu thêm. Đo trước, quyết định sau.

Đo 2 kích thước:
  - Ảnh gốc (nhỏ).
  - Resize về đúng độ phân giải camera (1280x720) — sát thực tế lúc chạy live.

Không camera, không mic, không UI. Chạy:
    cd ~/Desktop/skin_disease_rag
    python3 test_yolo_latency.py
"""

import time
import statistics
import numpy as np

import cv2
import torch
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH, YOLO_TO_DISEASE, DISEASE_VI_NAMES,
)

# Ngưỡng conf robot dùng để CHẤP NHẬN 1 detection (robot_main.py: DETECT_CONF_MIN)
CONF = 0.25          # dùng ngưỡng thấp khi ĐO để thấy cả detection yếu
CAMERA_W, CAMERA_H = 1280, 720
WARMUP = 5           # số lần chạy bỏ đi (CUDA autotune/biên dịch kernel)
MEASURE = 30         # số lần chạy để lấy thống kê

IMAGES = ["psoriasis.jpg", "leprosy.jpg"]


def extract_dets(results):
    """Trích detection y hệt _run_yolo() trong robot_main.py."""
    dets = []
    for r in results:
        for box in r.boxes:
            cls_id      = int(box.cls[0])
            cls_name    = r.names[cls_id]
            conf        = float(box.conf[0])
            disease_key = YOLO_TO_DISEASE.get(cls_name.lower(), cls_name.lower())
            vi_name     = DISEASE_VI_NAMES.get(disease_key, cls_name)
            dets.append({"vi": vi_name, "cls": cls_name, "conf": conf})
    return sorted(dets, key=lambda d: d["conf"], reverse=True)


def bench(model, frame, label):
    """Warmup rồi đo MEASURE lần, in thống kê ms + FPS."""
    for _ in range(WARMUP):
        model(frame, conf=CONF, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(MEASURE):
        t = time.time()
        results = model(frame, conf=CONF, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t) * 1000)

    ms_mean = statistics.mean(times)
    ms_min, ms_max = min(times), max(times)
    fps = 1000 / ms_mean
    print(f"    {label:22} {ms_mean:6.1f}ms TB  (min {ms_min:.1f} / max {ms_max:.1f})  ≈ {fps:.1f} FPS")
    return ms_mean


# ── Nạp model ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[YOLO] Device: {device}", flush=True)
if device == "cpu":
    print("[YOLO] ⚠️  CUDA không khả dụng — đang đo trên CPU.")

t0 = time.time()
model = YOLO(str(YOLO_MODEL_PATH))
print(f"[YOLO] Nạp model {time.time() - t0:.1f}s | classes: {list(model.names.values())}")
print("=" * 70)

# ── Chạy từng ảnh ─────────────────────────────────────────────────────────────
all_native = []
all_camera = []

for img_name in IMAGES:
    frame = cv2.imread(img_name)      # BGR numpy — GIỐNG frame từ camera
    if frame is None:
        print(f"\n❌ Không đọc được ảnh: {img_name} — bỏ qua.")
        continue

    h, w = frame.shape[:2]
    frame_cam = cv2.resize(frame, (CAMERA_W, CAMERA_H))

    print(f"\n📷 {img_name}  (gốc {w}x{h})")

    # Model detect gì?
    dets = extract_dets(model(frame, conf=CONF, verbose=False))
    if dets:
        top = dets[0]
        print(f"    → Detect: {top['vi']} ({top['cls']}), conf={top['conf']:.0%}"
              + (f"  [+{len(dets)-1} vùng khác]" if len(dets) > 1 else ""))
    else:
        print(f"    → Không detect được gì ở conf≥{CONF:.0%}")

    # Đo tốc độ 2 kích thước
    all_native.append(bench(model, frame,     f"gốc {w}x{h}:"))
    all_camera.append(bench(model, frame_cam, f"camera {CAMERA_W}x{CAMERA_H}:"))

# ── Kết luận ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if all_camera:
    avg_cam = statistics.mean(all_camera)
    print(f"📊 YOLO ở độ phân giải camera ({CAMERA_W}x{CAMERA_H}): "
          f"TB {avg_cam:.0f}ms/frame ≈ {1000/avg_cam:.1f} FPS trên {device.upper()}")
    print("\nDiễn giải cho quyết định kiosk:")
    print(f"  • Robot chạy YOLO mỗi ~0.3s/lần trong vòng viewfinder (time.sleep(0.3)).")
    if avg_cam < 300:
        print(f"  • {avg_cam:.0f}ms << 300ms → YOLO KHÔNG phải nút cổ chai. GPU đã quá đủ.")
        print("    → Không cần tối ưu YOLO thêm. Vấn đề GPU coi như khép lại.")
    else:
        print(f"  • {avg_cam:.0f}ms — cân nhắc: giảm imgsz, hoặc export TensorRT/FP16.")
print("✅ GIAI ĐOẠN 4a xong: đã có số liệu YOLO thật.")
