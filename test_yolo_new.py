"""
test_yolo_new.py — Test YOLO MỚI (5 bệnh) độc lập: không RAG, không DB, không voice.

Kiểm tra 3 thứ:
  1. Fork ultralytics_custom nạp được best.pt (kiến trúc mới)
  2. Tên class đúng: HSV, eczema, impetigo, lepra, ringworm → map sang tiếng Việt
  3. Detect + tốc độ trên đầu vào THẬT (camera CSI) hoặc ảnh tĩnh

Chạy:
    python3 test_yolo_new.py                 # camera CSI trực tiếp (mặc định)
    python3 test_yolo_new.py --image a.jpg   # test trên 1 ảnh
    python3 test_yolo_new.py --conf 0.15     # hạ ngưỡng nếu khó detect

Trong chế độ camera: cửa sổ hiện khung + nhãn bệnh. Nhấn 'q' hoặc ESC để thoát.
"""

import argparse
import os
import sys
import time

# KHÔNG dùng ultralytics_custom: fork tùy biến gây treo lúc suy luận.
# Weights baseline chạy tốt với ultralytics sẵn có.
print("[Setup] Dùng ultralytics sẵn có (không dùng fork ultralytics_custom)")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics
print(f"[Setup] ultralytics: {ultralytics.__file__}")

from config import YOLO_MODEL_PATH, YOLO_TO_DISEASE, DISEASE_VI_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--image", help="Test trên 1 ảnh thay vì camera")
ap.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
args = ap.parse_args()

# ── Nạp model ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[YOLO] Device: {device}")
print(f"[YOLO] Model:  {YOLO_MODEL_PATH}")

t0 = time.time()
model = YOLO(str(YOLO_MODEL_PATH))
print(f"[YOLO] ✅ Nạp model {time.time()-t0:.1f}s")
print(f"[YOLO] Classes: {model.names}")

# ── Kiểm tra mapping class → tiếng Việt ───────────────────────────────────────
print("\n=== Kiểm tra mapping (class YOLO → key → tên tiếng Việt) ===")
ok_map = True
for cls_name in model.names.values():
    key = YOLO_TO_DISEASE.get(cls_name.lower())
    vi  = DISEASE_VI_NAMES.get(key) if key else None
    if key and vi:
        print(f"  ✓ {cls_name:10} → {key:9} → {vi}")
    else:
        print(f"  ✗ {cls_name:10} → KHÔNG map được! Sửa YOLO_TO_DISEASE trong config.py")
        ok_map = False
if not ok_map:
    sys.exit("\n❌ Mapping sai — sửa config.py trước khi chạy robot.")
print("✅ Mapping OK\n")


def detections(frame, conf):
    """Trả list dict đã map sang tiếng Việt, sắp theo confidence giảm dần."""
    results = model(frame, conf=conf, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls[0])]
            key = YOLO_TO_DISEASE.get(cls_name.lower(), cls_name.lower())
            dets.append({
                "cls":  cls_name,
                "key":  key,
                "vi":   DISEASE_VI_NAMES.get(key, cls_name),
                "conf": float(box.conf[0]),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
            })
    return sorted(dets, key=lambda d: d["conf"], reverse=True)


# ── Chế độ ẢNH TĨNH ───────────────────────────────────────────────────────────
if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"❌ Không đọc được ảnh: {args.image}")
    h, w = frame.shape[:2]
    print(f"📷 {args.image} ({w}x{h}), conf≥{args.conf:.0%}")

    t = time.time()
    dets = detections(frame, args.conf)
    print(f"   Thời gian: {(time.time()-t)*1000:.0f}ms")

    if not dets:
        print("   → Không detect được gì. Thử hạ ngưỡng: --conf 0.10")
    else:
        for d in dets:
            print(f"   → {d['vi']} ({d['cls']}) conf={d['conf']:.0%}")
    sys.exit(0)


# ── Chế độ CAMERA CSI ─────────────────────────────────────────────────────────
from csi_camera import CSICamera

print("📹 Mở camera CSI... (chĩa vào vùng da/ảnh bệnh; 'q' hoặc ESC để thoát)")
cam = CSICamera(sensor_id=0, width=1280, height=720)
time.sleep(2)   # camera warm-up

times, n_frames, last_print = [], 0, time.time()
try:
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        t = time.time()
        dets = detections(frame, args.conf)
        dt = (time.time() - t) * 1000
        times.append(dt)
        n_frames += 1

        # Vẽ khung + nhãn
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['vi']} {d['conf']:.0%}"
            cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        avg = sum(times[-30:]) / len(times[-30:])
        cv2.putText(frame, f"{avg:.0f}ms | {1000/avg:.1f} FPS | {device}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("YOLO test — 5 benh moi", frame)

        # In ra terminal mỗi giây (để đọc cả khi không nhìn cửa sổ)
        if time.time() - last_print > 1.0:
            if dets:
                top = dets[0]
                print(f"  → {top['vi']} ({top['cls']}) {top['conf']:.0%}"
                      f"  [{avg:.0f}ms]")
            else:
                print(f"  → (không detect)  [{avg:.0f}ms]")
            last_print = time.time()

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
    if times:
        avg = sum(times) / len(times)
        print(f"\n📊 {n_frames} frames | TB {avg:.0f}ms/frame ≈ {1000/avg:.1f} FPS trên {device.upper()}")
    print("✅ Test YOLO xong.")
