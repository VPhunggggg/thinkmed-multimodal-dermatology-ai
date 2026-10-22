"""
test_camera_quality.py — So sánh trực quan các preset ISP khác nhau.

Chụp lần lượt từng preset, lưu file .jpg riêng để mở và so sánh bằng mắt.
KHÔNG có cách nào chọn preset "đúng" từ xa mà không xem ảnh thật — script
này chỉ tạo ra các lựa chọn để bạn tự so sánh trên hardware/ánh sáng
thực tế của bạn.

Usage:
    python test_camera_quality.py
    # Mở các file /home/fptuadmin/quality_*.jpg để so sánh
"""

import time
import cv2

from csi_camera import CSICamera, DEFAULT_SKIN_ISP

PRESETS = {
    # Không set gì — baseline để so sánh, dùng auto mặc định của gstreamer
    "off": {},

    # Preset mặc định đang dùng trong robot_main.py
    "skin_macro_default": DEFAULT_SKIN_ISP,

    # Ít khử nhiễu hơn, sharpen mạnh hơn — nếu ảnh gốc bị mờ/nhòe
    "sharper": {
        "wbmode": 1, "saturation": 1.0,
        "tnr-mode": 2, "tnr-strength": 0.1,
        "ee-mode": 2, "ee-strength": 0.5,
    },

    # Khử nhiễu mạnh hơn — nếu ảnh gốc bị noise/hạt (thường do thiếu sáng)
    "less_noise": {
        "wbmode": 1, "saturation": 1.0,
        "tnr-mode": 2, "tnr-strength": 0.5,
        "ee-mode": 1, "ee-strength": 0.2,
    },
}


def capture_preset(name: str, isp: dict) -> bool:
    print(f"\n=== Preset: {name} ===")
    print(f"    ISP: {isp or '(mặc định gstreamer, không set gì)'}")

    try:
        cam = CSICamera(sensor_id=0, width=1280, height=720, isp_settings=isp)
    except FileNotFoundError:
        print("❌ Không tìm thấy gst-launch-1.0")
        return False

    if not cam.isOpened():
        print("❌ Không mở được camera với preset này:")
        print(cam.get_stderr_tail())
        return False

    # Đọc vài frame, lấy frame cuối (ổn định hơn frame đầu do exposure
    # vừa hội tụ)
    frame = None
    for _ in range(15):
        ok, f = cam.read()
        if ok:
            frame = f
        time.sleep(0.1)

    cam.release()

    if frame is None:
        print("❌ Không đọc được frame nào")
        return False

    path = f"/home/fptuadmin/quality_{name}.jpg"
    cv2.imwrite(path, frame)
    print(f"✅ Đã lưu: {path}")
    return True


def main():
    print("So sánh chất lượng ảnh với các preset ISP khác nhau.")
    print("Giữ NGUYÊN vùng chụp (vd: đưa tay/vùng da mẫu vào khung) trong")
    print("suốt quá trình để so sánh công bằng giữa các preset.\n")
    input("Nhấn Enter khi đã sẵn sàng, giữ nguyên vị trí trong ~30 giây...")

    for name, isp in PRESETS.items():
        capture_preset(name, isp)
        time.sleep(0.5)   # nghỉ giữa các lần restart pipeline

    print("\n" + "=" * 50)
    print("Mở các file /home/fptuadmin/quality_*.jpg để so sánh.")
    print("Preset nào đẹp nhất → sửa DEFAULT_SKIN_ISP trong csi_camera.py")
    print("hoặc truyền isp_settings={...} tương ứng khi tạo CSICamera().")
    print("=" * 50)


if __name__ == "__main__":
    main()
