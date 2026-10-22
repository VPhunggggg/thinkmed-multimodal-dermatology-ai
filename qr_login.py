"""
qr_login.py — Module Quét mã QR & Nhận diện người dùng cho Robot ThinkMed.

Tính năng:
  - Nhận diện 3 người dùng (code cứng theo yêu cầu):
      + ID001: Nguyễn Việt Phương (Đội trưởng)
      + ID002: Nguyễn Nhật Long
      + ID003: Nguyễn Hoàng Huy
  - Tích hợp cv2.QRCodeDetector có sẵn trong OpenCV (siêu tốc, không cần cài thêm thư viện).
  - Tự động chào bằng giọng nói tiếng Việt qua audio.tts.speak (hoặc fallback).
  - Có thể chạy độc lập để test webcam: python qr_login.py
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

# Danh sách 3 người dùng code cứng
QR_USERS = {
    "ID001": {
        "id": "ID001",
        "name": "Nguyễn Việt Phương",
        "role": "Bệnh nhân / Đội trưởng",
        "gender": "bạn",
        "age": 21,
        "greeting": "Chào bạn Nguyễn Việt Phương! Chúc bạn một ngày tốt lành.",
        "short_name": "Phương",
    },
    "ID002": {
        "id": "ID002",
        "name": "Nguyễn Nhật Long",
        "role": "Bệnh nhân",
        "gender": "bạn",
        "age": 20,
        "greeting": "Chào bạn Nguyễn Nhật Long! Chúc bạn một ngày tốt lành.",
        "short_name": "Long",
    },
    "ID003": {
        "id": "ID003",
        "name": "Nguyễn Hoàng Huy",
        "role": "Bệnh nhân",
        "gender": "bạn",
        "age": 27,
        "greeting": "Chào bạn Nguyễn Hoàng Huy! Chúc bạn một ngày tốt lành.",
        "short_name": "Huy",
    },
}


def lookup_user(qr_raw_data: str):
    """
    Tra cứu thông tin người dùng từ dữ liệu mã QR.
    Hỗ trợ linh hoạt:
      - Mã chính xác: 'ID001', 'ID002', 'ID003'
      - Chứa ID trong chuỗi (URL, JSON, văn bản): 'THINKMED:ID001', 'https://...ID001'
      - Chứa tên không dấu hoặc có dấu
    """
    if not qr_raw_data:
        return None

    raw_clean = qr_raw_data.strip()
    raw_upper = raw_clean.upper()

    # 1. Khớp mã ID trực tiếp
    if raw_upper in QR_USERS:
        return QR_USERS[raw_upper]

    # 2. Khớp mã ID con trong chuỗi
    for uid, user in QR_USERS.items():
        if uid in raw_upper:
            return user

    # 3. Khớp theo tên tiếng Việt hoặc tên không dấu
    raw_lower = raw_clean.lower()
    for uid, user in QR_USERS.items():
        name_lower = user["name"].lower()
        if name_lower in raw_lower or user["short_name"].lower() in raw_lower:
            return user

    return None


def is_phuong_user(user) -> bool:
    """
    Kiểm tra xem người dùng có phải là anh Nguyễn Việt Phương (ID001 - Đội trưởng) hay không.
    Hỗ trợ kiểm tra qua user dict, ID001, hoặc chuỗi tên.
    """
    if not user:
        return False
    if isinstance(user, str):
        u_str = user.strip().lower()
        return "id001" in u_str or "phương" in u_str or "phuong" in u_str

    if isinstance(user, dict):
        uid = str(user.get("id", "")).strip().upper()
        if uid == "ID001":
            return True
        name = str(user.get("name", "")).lower()
        short = str(user.get("short_name", "")).lower()
        return "phương" in name or "phuong" in name or "phương" in short or "phuong" in short

    return False


class QRScanner:
    """
    Bộ giải mã QR đa tầng thông minh:
      1. Ưu tiên pyzbar (siêu tốc, độc lập, không phụ thuộc QUIRC của OpenCV, hoạt động hoàn hảo trên Jetson / Linux / Windows).
      2. Fallback sang cv2.QRCodeDetector nếu chưa cài pyzbar.
    """

    def __init__(self):
        self.use_pyzbar = False
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            self.pyzbar_decode = pyzbar_decode
            self.use_pyzbar = True
            print("[QRScanner] ✅ Đã kích hoạt bộ giải mã pyzbar (hoạt động độc lập, không cần QUIRC).", flush=True)
        except Exception:
            self.pyzbar_decode = None
            print("[QRScanner] ⚠️ Chưa cài pyzbar (trên Linux/Jetson chạy: sudo apt install libzbar0 && pip install pyzbar). Fallback sang cv2.", flush=True)

        self.cv_detector = None
        if not self.use_pyzbar:
            try:
                self.cv_detector = cv2.QRCodeDetector()
            except Exception:
                pass
        self.last_detected_id = None
        self.last_detected_time = 0.0

    def scan_frame(self, frame: np.ndarray):
        """
        Quét mã QR từ 1 khung hình camera.
        Trả về: (user_info, raw_text, points)
          - user_info: dict thông tin người dùng (hoặc None nếu không khớp)
          - raw_text: nội dung chuỗi giải mã từ QR (chuỗi rỗng nếu chưa thấy)
          - points: tọa độ 4 đỉnh của QR trên ảnh (hoặc None)
        """
        if frame is None or frame.size == 0:
            return None, "", None

        # 1. Nếu có pyzbar: DÙNG ĐỘC LẬP 100%, KHÔNG BAO GIỜ GỌI OPENCV QUIRC!
        if self.use_pyzbar and self.pyzbar_decode is not None:
            try:
                decoded_objs = self.pyzbar_decode(frame)
                for obj in decoded_objs:
                    raw_text = obj.data.decode("utf-8", errors="ignore").strip()
                    if raw_text:
                        pts = None
                        if obj.polygon and len(obj.polygon) >= 4:
                            pts = np.array([(p.x, p.y) for p in obj.polygon], dtype=np.float32)
                        elif obj.rect:
                            r = obj.rect
                            pts = np.array([
                                [r.left, r.top],
                                [r.left + r.width, r.top],
                                [r.left + r.width, r.top + r.height],
                                [r.left, r.top + r.height]
                            ], dtype=np.float32)
                        user = lookup_user(raw_text)
                        return user, raw_text, pts
            except Exception:
                pass
            # pyzbar đã quét xong (chưa thấy mã trong frame này) -> Trả về None ngay, KHÔNG fallback OpenCV!
            return None, "", None

        # 2. Chỉ dùng cv2.QRCodeDetector khi MÁY CHƯA CÀI PYZBAR
        if self.cv_detector is not None:
            try:
                val, pts, _ = self.cv_detector.detectAndDecode(frame)
                if val and len(val.strip()) > 0:
                    raw_text = val.strip()
                    user = lookup_user(raw_text)
                    return user, raw_text, pts
            except Exception:
                self.cv_detector = None  # Tắt luôn nếu lỗi để không spam log

        return None, "", None


# ── Hàm chạy độc lập để test với Camera máy tính / Jetson ─────────────────────

def test_standalone_camera(camera_index=0):
    """Chạy thử nghiệm giao diện quét mã QR độc lập với webcam."""
    print("=" * 60)
    print("🤖 THINKMED - CHƯƠNG TRÌNH TEST QUÉT MÃ QR NHẬN DIỆN NGƯỜI DÙNG")
    print("=" * 60)
    print("Danh sách 3 người dùng đã đăng ký:")
    for uid, u in QR_USERS.items():
        print(f"  • {uid}: {u['name']} ({u['role']})")
    print("\nKhởi động Camera (Nhấn ESC hoặc Q để thoát)...")

    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Không mở được camera index {camera_index}")
        return

    scanner = QRScanner()
    recognized_user = None
    welcome_time = 0.0

    # Cố gắng import TTS
    speak_fn = None
    try:
        from audio.tts import speak
        speak_fn = speak
    except Exception:
        pass

    window_name = "ThinkMed - Test Quet Ma QR"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 600)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]
            display = frame.copy()

            # Quét mã QR
            user, raw_text, pts = scanner.scan_frame(frame)

            # Vẽ bounding box quanh mã QR nếu tìm thấy
            if pts is not None and len(pts) > 0:
                pts_int = pts.astype(int)
                cv2.polylines(display, [pts_int], True, (0, 255, 0), 3)

            now = time.time()
            if user is not None:
                if recognized_user != user or (now - welcome_time > 5.0):
                    recognized_user = user
                    welcome_time = now
                    print(f"\n✅ ĐÃ NHẬN DIỆN MÃ: {user['id']} -> {user['name']}")
                    msg = f"Chào bạn {user['name']}! Chào mừng bạn đến với hệ thống ThinkMed."
                    print(f"🤖 Robot: {msg}")
                    if speak_fn:
                        import threading
                        threading.Thread(target=speak_fn, args=(msg,), daemon=True).start()

            # Hiển thị thông báo trên màn hình
            if recognized_user and (now - welcome_time < 4.0):
                # Banner màu xanh lá thông báo đăng nhập thành công
                cv2.rectangle(display, (40, 40), (w - 40, 160), (0, 180, 50), -1)
                cv2.rectangle(display, (40, 40), (w - 40, 160), (255, 255, 255), 2)
                cv2.putText(display, f"DANG NHAP THANH CONG: {recognized_user['id']}", 
                            (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(display, f"Ho va ten: {recognized_user['name']} ({recognized_user['role']})", 
                            (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                # Khung ngắm quét mã QR ở giữa màn hình
                box_size = 280
                cx, cy = w // 2, h // 2
                x1, y1 = cx - box_size // 2, cy - box_size // 2
                x2, y2 = cx + box_size // 2, cy + box_size // 2

                # Vẽ 4 góc khung quét
                corner_len = 35
                color = (255, 180, 0)
                # Góc trên trái
                cv2.line(display, (x1, y1), (x1 + corner_len, y1), color, 4)
                cv2.line(display, (x1, y1), (x1, y1 + corner_len), color, 4)
                # Góc trên phải
                cv2.line(display, (x2, y1), (x2 - corner_len, y1), color, 4)
                cv2.line(display, (x2, y1), (x2, y1 + corner_len), color, 4)
                # Góc dưới trái
                cv2.line(display, (x1, y2), (x1 + corner_len, y2), color, 4)
                cv2.line(display, (x1, y2), (x1, y2 - corner_len), color, 4)
                # Góc dưới phải
                cv2.line(display, (x2, y2), (x2 - corner_len, y2), color, 4)
                cv2.line(display, (x2, y2), (x2, y2 - corner_len), color, 4)

                # Dòng quét laser di chuyển
                scan_y = int(y1 + ((np.sin(now * 4) + 1) / 2) * box_size)
                cv2.line(display, (x1 + 5, scan_y), (x2 - 5, scan_y), (0, 230, 255), 2)

                cv2.putText(display, "HUONG MA QR VAO KHUNG DE DANG NHAP", 
                            (cx - 240, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_id = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_id = int(sys.argv[1])
    test_standalone_camera(cam_id)
