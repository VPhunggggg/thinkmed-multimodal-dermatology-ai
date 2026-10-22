"""
test_csi_camera.py — Test class CSICamera (subprocess gst-launch-1.0).

Không cần cv2 GStreamer, không cần load RAG/YOLO — test nhanh, riêng biệt.

Usage:
    python test_csi_camera.py
"""

import sys
import time

from csi_camera import CSICamera

print("Khởi động CSICamera (nvarguscamerasrc qua subprocess)...")
print("Chờ warmup ~1.5s...\n")

try:
    cam = CSICamera(sensor_id=0, width=1280, height=720, fps=30)
except FileNotFoundError:
    print("❌ Không tìm thấy lệnh 'gst-launch-1.0'.")
    print("   Cài: sudo apt install -y gstreamer1.0-tools")
    sys.exit(1)

if not cam.isOpened():
    print("❌ gst-launch-1.0 process đã thoát ngay lập tức.")
    print("Log lỗi:")
    print(cam.get_stderr_tail())
    sys.exit(1)

print("Đang đọc frame (thử tối đa 5 giây)...")
frame = None
t0 = time.time()
while time.time() - t0 < 5:
    ok, f = cam.read()
    if ok:
        frame = f
        break
    time.sleep(0.1)

if frame is None:
    print("❌ Không đọc được frame nào trong 5s.")
    print("Log lỗi từ gst-launch-1.0:")
    print(cam.get_stderr_tail())
    cam.release()
    sys.exit(1)

h, w = frame.shape[:2]
b = float(frame[:, :, 0].mean())
g = float(frame[:, :, 1].mean())
r = float(frame[:, :, 2].mean())

print(f"\n✅ Frame nhận được: {w}x{h}")
print(f"   B={b:.0f}  G={g:.0f}  R={r:.0f}")

if g > r * 1.8 and g > b * 1.8:
    print("⚠️  Kênh xanh lá vẫn cao bất thường — có thể vẫn là raw Bayer.")
else:
    print("✅ Màu cân bằng — debayer đã hoạt động đúng!")

import cv2
out_path = "/home/fptuadmin/test_csi_camera.jpg"
# Lưu ý: dùng cv2 ở đây chỉ để ghi JPG — cv2.imwrite() không đụng tới
# module GStreamer nên bản pip (không GStreamer) vẫn dùng bình thường được.
cv2.imwrite(out_path, frame)
print(f"Đã lưu: {out_path}")

# Đọc thêm vài frame liên tiếp để kiểm tra ổn định
print("\nĐọc thêm 10 frame liên tiếp để kiểm tra tốc độ...")
t0 = time.time()
count = 0
for _ in range(10):
    ok, f = cam.read()
    if ok:
        count += 1
    time.sleep(0.05)
elapsed = time.time() - t0
print(f"Đọc {count}/10 frame trong {elapsed:.2f}s")

cam.release()
print("\n✅ Test hoàn tất.")
