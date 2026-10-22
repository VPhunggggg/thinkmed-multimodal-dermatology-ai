"""
diag_camera_yolo.py — Vì sao ảnh tĩnh detect được mà camera thì không?

Chụp frame THẬT từ camera CSI, rồi thử YOLO trên nhiều biến thể để tìm nguyên nhân:
  A. Frame nguyên bản (như code đang dùng)
  B. Frame ĐẢO KÊNH MÀU (BGR↔RGB)  ← nghi phạm số 1
  C. Frame nguyên bản, ngưỡng cực thấp (xem model có tín hiệu gì không)

Lưu các frame ra file để bạn MỞ RA XEM tận mắt (màu có đúng không, có nét không).

Chạy (chĩa camera vào ảnh bệnh trên điện thoại, giữ yên):
    python3 diag_camera_yolo.py
"""

import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import YOLO_MODEL_PATH, YOLO_TO_DISEASE, DISEASE_VI_NAMES
from csi_camera import CSICamera

print(f"[YOLO] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
model = YOLO(str(YOLO_MODEL_PATH))
print(f"[YOLO] classes: {model.names}\n")


def run(frame, conf, tag):
    """Chạy YOLO, in kết quả."""
    t = time.time()
    results = model(frame, conf=conf, verbose=False)
    dt = (time.time() - t) * 1000
    dets = []
    for r in results:
        for b in r.boxes:
            cls = r.names[int(b.cls[0])]
            key = YOLO_TO_DISEASE.get(cls.lower(), cls.lower())
            dets.append((DISEASE_VI_NAMES.get(key, cls), cls, float(b.conf[0])))
    dets.sort(key=lambda d: d[2], reverse=True)

    if dets:
        top = dets[0]
        print(f"  {tag:38} → {top[1]:9} {top[2]:.1%}   ({dt:.0f}ms, {len(dets)} box)")
    else:
        print(f"  {tag:38} → (không có box)         ({dt:.0f}ms)")
    return dets


# ── Chụp frame thật từ camera ────────────────────────────────────────────────
print("📹 Mở camera... chĩa vào ảnh bệnh, GIỮ YÊN 5 giây.")
cam = CSICamera(sensor_id=0, width=1280, height=720)
time.sleep(3)   # warm-up + để bạn kịp chĩa

frame = None
for _ in range(30):          # đọc vài frame, lấy frame cuối (ổn định nhất)
    ok, f = cam.read()
    if ok and f is not None:
        frame = f
    time.sleep(0.05)
cam.release()

if frame is None:
    raise SystemExit("❌ Không đọc được frame nào từ camera.")

print(f"✅ Đã chụp frame: shape={frame.shape}, dtype={frame.dtype}")
print(f"   Giá trị pixel: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

# Kênh màu trung bình — giúp phát hiện đảo kênh
b, g, r = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
print(f"   Trung bình kênh [0]={b:.1f} [1]={g:.1f} [2]={r:.1f}")
print("   (Da người: kênh ĐỎ thường CAO NHẤT. Nếu kênh [0] cao nhất → có thể đang là RGB, tức bị đảo.)\n")

# ── Lưu ra file để xem tận mắt ───────────────────────────────────────────────
cv2.imwrite("diag_frame_original.jpg", frame)
frame_swapped = frame[:, :, ::-1].copy()      # đảo kênh BGR↔RGB
cv2.imwrite("diag_frame_swapped.jpg", frame_swapped)
print("💾 Đã lưu: diag_frame_original.jpg + diag_frame_swapped.jpg")
print("   → MỞ 2 ẢNH NÀY RA XEM: cái nào MÀU DA TRÔNG ĐÚNG (hồng/nâu, không ám xanh)?\n")

# ── Chạy YOLO trên các biến thể ──────────────────────────────────────────────
print("=== KẾT QUẢ YOLO ===")
d_orig = run(frame,          0.25, "A. Frame nguyên bản (conf 25%)")
d_swap = run(frame_swapped,  0.25, "B. Frame ĐẢO KÊNH màu (conf 25%)")
print()
run(frame,         0.01, "C. Nguyên bản, ngưỡng 1% (dò tín hiệu)")
run(frame_swapped, 0.01, "D. Đảo kênh, ngưỡng 1%")

# ── Kết luận ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
if d_swap and not d_orig:
    print("🎯 NGUYÊN NHÂN: KÊNH MÀU BỊ ĐẢO.")
    print("   Camera trả RGB nhưng code coi là BGR → model thấy màu da sai.")
    print("   FIX: đảo kênh frame trong csi_camera.py trước khi trả về.")
elif d_orig and not d_swap:
    print("✅ Kênh màu ĐÚNG (nguyên bản detect được, đảo kênh thì không).")
    print("   → Nếu robot vẫn không detect: vấn đề ở nội dung frame lúc đó")
    print("     (mờ/xa/thiếu sáng) hoặc ngưỡng. Xem 2 ảnh đã lưu.")
elif d_orig and d_swap:
    print("✅ Cả hai đều detect được → kênh màu không phải vấn đề.")
else:
    print("⚠️  CẢ HAI đều KHÔNG detect được.")
    print("   → Model không nhận ra thứ trong khung hình. Mở diag_frame_original.jpg:")
    print("     ảnh có rõ nét không? vùng bệnh có đủ lớn/đủ sáng không?")
    print("     Thử: đưa camera gần hơn, tăng sáng, hoặc chĩa vào ảnh bệnh rõ nét hơn.")
print("=" * 66)
