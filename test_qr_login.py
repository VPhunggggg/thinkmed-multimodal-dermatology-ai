"""
test_qr_login.py — Script kiểm thử Giao diện Đăng nhập Quét mã QR cho ThinkMed.

Không yêu cầu mô hình YOLO, RAG hay GPIO — chỉ cần webcam và giao diện UI để kiểm tra:
  - Khung ngắm quét mã QR với hiệu ứng laser và 4 góc neon.
  - Tự động nhận diện 3 người dùng code cứng:
      + ID001: Nguyễn Việt Phương
      + ID002: Nguyễn Nhật Long
      + ID003: Nguyễn Hoàng Huy
  - Robot đọc lời chào mừng bằng giọng nói tiếng Việt qua TTS.
  - Hiển thị thẻ xác thực danh tính thành công.
  - Chuyển sang màn hình chờ IDLE và gắn huy hiệu định danh người dùng.

Chạy:
    python test_qr_login.py           # Dùng webcam 0
    python test_qr_login.py 1         # Dùng webcam 1 (nếu có webcam rời)
"""

import os
import sys
import time
import threading
import cv2

# Cấu hình encoding console UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from robot_ui import ThinkMedUI, State
from qr_login import QRScanner, QR_USERS


def _worker_test(ui: ThinkMedUI, cam_id: int):
    print("\n[Worker] Khởi tạo camera để kiểm thử quét mã QR...", flush=True)
    if os.name == "nt":
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"⚠️ Không mở được camera {cam_id}. Vui lòng kiểm tra lại webcam!", flush=True)
        ui.stop()
        return

    # Thử nạp TTS nếu có
    speak_fn = None
    try:
        from audio.tts import speak
        speak_fn = speak
        print("✅ Đã kết nối module giọng nói TTS.", flush=True)
    except Exception as e:
        print(f"ℹ️ Không có TTS ({e}) — sẽ in lời thoại ra màn hình console.", flush=True)

    scanner = QRScanner()
    print("\n[Worker] Sẵn sàng! Đang mở Giao diện Đăng nhập Quét mã QR...", flush=True)
    print("Mẹo: Mở file 'assets/qr_codes/qr_all_3_members.png' trên điện thoại và đưa vào trước camera.")

    # Bắt đầu ngay với giao diện quét QR
    ui.set_state(State.QR_LOGIN, qr_user=None, qr_pts=None, success=False)

    try:
        while ui._running:
            # 1. Chế độ Quét QR Đăng nhập
            if ui.state == State.QR_LOGIN:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                user, raw_text, pts = scanner.scan_frame(frame)
                ui.update_viewfinder(frame)

                if user is not None:
                    print(f"\n🎯 [QR] ĐÃ NHẬN DIỆN MÃ: {user['id']} -> {user['name']}", flush=True)
                    ui.logged_in_user = user
                    ui.set_state(State.QR_LOGIN, qr_user=user, qr_pts=pts, success=True)

                    greeting = f"Chào bạn {user['name']}! Chúc bạn một ngày tốt lành. Mời bạn chọn khám da hoặc tư vấn."
                    print(f"🤖 Robot phát biểu: \"{greeting}\"\n", flush=True)

                    if speak_fn:
                        threading.Thread(target=speak_fn, args=(greeting,), daemon=True).start()

                    time.sleep(2.5)
                    ui.set_state(State.IDLE)
                time.sleep(0.03)

            # 2. Chế độ Màn hình chờ IDLE
            elif ui.state == State.IDLE:
                # Đọc camera liên tục để frame sẵn sàng khi chuyển state
                ret, frame = cap.read()
                if ret:
                    ui.update_viewfinder(frame)

                # Kiểm tra nếu người dùng nhấn vào nút Quét QR / Đổi người trên màn hình IDLE
                if ui.tap_event.is_set():
                    action = ui.selected_action
                    ui.tap_event.clear()
                    if action == "qr_login":
                        print("\n[Worker] 🔄 Người dùng nhấn quét lại mã QR...", flush=True)
                        ui.set_state(State.QR_LOGIN, qr_user=None, qr_pts=None, success=False)
                    elif action == "kham_da":
                        user_name = ui.logged_in_user['name'] if ui.logged_in_user else "bạn"
                        print(f"\n[Worker] 🩺 Bấm 'Khám da' (Đã đăng nhập: {user_name})", flush=True)
                        ui.set_state(State.GREETING)
                        msg = f"Xin chào {user_name}! Đang chuẩn bị camera khám da."
                        if speak_fn:
                            threading.Thread(target=speak_fn, args=(msg,), daemon=True).start()
                        time.sleep(2.0)
                        ui.set_state(State.IDLE)
                    elif action == "tu_van":
                        user_name = ui.logged_in_user['name'] if ui.logged_in_user else "bạn"
                        print(f"\n[Worker] 💬 Bấm 'Tư vấn' (Đã đăng nhập: {user_name})", flush=True)
                        ui.set_state(State.GREETING)
                        msg = f"Xin chào {user_name}! Mời bạn đặt câu hỏi tư vấn da liễu."
                        if speak_fn:
                            threading.Thread(target=speak_fn, args=(msg,), daemon=True).start()
                        time.sleep(2.0)
                        ui.set_state(State.IDLE)

                time.sleep(0.04)

            # 3. Các state khác (nếu bấm thoát từ QR_LOGIN)
            else:
                if ui.exit_event.is_set():
                    ui.exit_event.clear()
                    ui.set_state(State.IDLE)
                time.sleep(0.04)

    finally:
        cap.release()
        print("[Worker] Đã giải phóng camera an toàn.", flush=True)


def main():
    cam_id = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_id = int(sys.argv[1])

    print("=" * 65)
    print("🤖 THINKMED - GIAO DIỆN KIỂM THỬ QUÉT MÃ QR NHẬN DIỆN NGƯỜI DÙNG")
    print("=" * 65)
    print("3 người dùng trong hệ thống:")
    for uid, u in QR_USERS.items():
        print(f"   [{uid}] : {u['name']} ({u['role']})")
    print("-" * 65)
    print("Điều khiển:")
    print("   • Đưa mã QR vào trước webcam để robot quét và đọc chào mừng.")
    print("   • Nhấn nút [X Bỏ qua] ở góc trên phải để vào màn hình chính.")
    print("   • Nhấn vào huy hiệu người dùng góc trên trái để quét đổi người.")
    print("   • Nhấn ESC hoặc Q trên bàn phím để tắt cửa sổ.")
    print("=" * 65)

    # Chạy giao diện ở chế độ cửa sổ để tiện quan sát
    ui = ThinkMedUI(fullscreen=False)
    worker = threading.Thread(target=_worker_test, args=(ui, cam_id), daemon=True)
    worker.start()

    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()
        worker.join(timeout=2.0)
        print("[Test] Đã hoàn tất phiên kiểm thử.")


if __name__ == "__main__":
    main()
