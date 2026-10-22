"""
robot_ui.py — Giao diện robot ThinkMed dùng OpenCV.

QUAN TRỌNG — Threading: OpenCV HighGUI (GTK backend) yêu cầu MỌI lệnh GUI
(namedWindow, imshow, waitKey, setWindowProperty, setMouseCallback) phải
chạy trên CÙNG MỘT thread — phải là thread gọi run(). __init__ KHÔNG được
gọi bất kỳ hàm cv2 GUI nào (chỉ set state); run() phải được gọi từ MAIN
THREAD của process (xem robot_main.py — ui.run() chạy main thread,
business logic chạy thread phụ).

States: IDLE, GREETING, VIEWFINDER, ANALYZING, RESULT_OK, RESULT_MULTI,
        RESULT_NONE, LISTENING, SPEAKING

Touch: touchscreen Linux map thành mouse event chuẩn, bắt qua
       cv2.setMouseCallback().

Cài: pip install opencv-python numpy Pillow
     sudo apt install -y fonts-dejavu-core
"""

# ── Windows DPI Awareness (Chống nhòe mờ giao diện trên màn hình 1080p / High-DPI) ──
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
except Exception:
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except Exception:
        pass

import math
import os
import sys
import subprocess
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


class State(Enum):
    IDLE         = auto()
    QR_LOGIN     = auto()
    GREETING     = auto()
    VIEWFINDER   = auto()
    ANALYZING    = auto()
    RESULT_OK    = auto()
    RESULT_MULTI = auto()
    RESULT_NONE  = auto()
    LISTENING    = auto()
    SPEAKING     = auto()


# ── Màu sắc (BGR) ─────────────────────────────────────────────────────────────

BG       = (25,  15,  15)
WHITE    = (255, 255, 255)
BLUE     = (255, 156,  64)
GREEN    = (142, 199,  72)
AMBER    = ( 64, 190, 255)
RED      = ( 74,  75, 234)
GRAY     = (140, 120, 120)
BRAND    = (255, 156,  64)

WINDOW_NAME = "ThinkMed"

# ── Nhạc nền IDLE: chỉnh ở đây ─────────────────────────────────────────────────
IDLE_VOLUME   = 2.0    # khuếch đại âm lượng (1.0 = gốc; 1.8 = to hơn ~80%; >3 dễ rè)
IDLE_FADE_IN  = 1.5    # giây fade-in khi bắt đầu phát (vào IDLE)


class ThinkMedUI:
    W, H = 1024, 600   # design resolution

    def __init__(self, fullscreen: bool = True):
        self.fullscreen  = fullscreen
        self.state       = State.IDLE
        self.frame_n     = 0
        self.result_img: Optional[np.ndarray] = None
        self.result_data: dict = {}
        self._lock       = threading.Lock()
        self._running      = True
        self.crashed       = False
        self.tap_event     = threading.Event()
        self.capture_event = threading.Event()
        self.exit_event    = threading.Event()
        self.audio_energy  = 0.0
        self.is_user_speaking = False
        self._font_cache   = {}
        self.selected_action = "kham_da"
        self.logged_in_user = None  # Thông tin người dùng đăng nhập qua QR code
        self.booking_event  = threading.Event()  # Sự kiện người dùng bấm nút Có/Không đặt lịch
        self.booking_choice = None               # 'yes' hoặc 'no'
        # Nhạc nền IDLE: phát lặp file idle_loop.mp3 (nếu có) khi ở trạng thái IDLE
        self._idle_audio_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "idle_loop.mp3")
        self._idle_proc = None

        # Ảnh nền giao diện IDLE mới (theo ảnh thực tế của robot ThinkMed)
        self._idle_bg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_idle_base.png")
        self._idle_blink_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_idle_blink.png")
        self._idle_bg = cv2.imread(self._idle_bg_path) if os.path.exists(self._idle_bg_path) else None
        self._idle_blink = cv2.imread(self._idle_blink_path) if os.path.exists(self._idle_blink_path) else None

        # Ảnh nền giao diện Thu âm / Lắng nghe mới (theo ảnh thiết kế ThinkMed)
        self._listening_bg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_listening_base.png")
        self._listening_bg = cv2.imread(self._listening_bg_path) if os.path.exists(self._listening_bg_path) else None
        self._smooth_energy = 0.0

        # Ảnh nền giao diện Chụp ảnh / Khung ngắm mới (theo ảnh thiết kế ThinkMed)
        self._viewfinder_base_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_viewfinder_base.png")
        self._viewfinder_mascot_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_viewfinder_mascot.png")
        self._viewfinder_base = cv2.imread(self._viewfinder_base_path) if os.path.exists(self._viewfinder_base_path) else None
        self._viewfinder_mascot = cv2.imread(self._viewfinder_mascot_path, cv2.IMREAD_UNCHANGED) if os.path.exists(self._viewfinder_mascot_path) else None

        # Ảnh nền giao diện Robot Trả Lời mới (theo ảnh thiết kế ThinkMed)
        self._speaking_bg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_speaking_base.png")
        self._speaking_bg = cv2.imread(self._speaking_bg_path) if os.path.exists(self._speaking_bg_path) else None

        # Ảnh nền giao diện Kết quả Phân tích mới (theo ảnh thiết kế ThinkMed)
        self._result_bg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_result_base.png")
        self._result_bg = cv2.imread(self._result_bg_path) if os.path.exists(self._result_bg_path) else None

        # Ảnh nền giao diện Không Phát Hiện Bệnh mới (theo ảnh thiết kế ThinkMed)
        self._result_none_bg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "ui_result_none_base.png")
        self._result_none_bg = cv2.imread(self._result_none_bg_path) if os.path.exists(self._result_none_bg_path) else None
        # KHÔNG gọi cv2.namedWindow/setWindowProperty/setMouseCallback ở đây
        # — phải nằm trong run() để cùng thread với imshow/waitKey.

    # ── Public API (gọi từ worker thread) ──────────────────────────────────────

    def update_audio_level(self, energy: float, is_speaking: bool):
        """Cập nhật năng lượng âm lượng thực tế của micro để vẽ sóng âm sống động theo mức độ nói to/nhỏ."""
        with self._lock:
            # Làm mượt năng lượng âm lượng: phản hồi nhanh khi âm lượng tăng, giảm từ tốn
            if energy > self._smooth_energy:
                self._smooth_energy = 0.35 * self._smooth_energy + 0.65 * energy
            else:
                self._smooth_energy = 0.82 * self._smooth_energy + 0.18 * energy
            self.audio_energy = self._smooth_energy
            self.is_user_speaking = is_speaking

    def set_state(self, state: State, **kwargs):
        # Nhạc nền: vào IDLE → phát lặp; rời IDLE → tắt ngay (tránh đè lên chào/hội thoại)
        if state == State.IDLE:
            self._start_idle_audio()
        else:
            self._stop_idle_audio()
        with self._lock:
            self.state       = state
            self.result_data = kwargs

    def _start_idle_audio(self):
    	
        """Phát lặp idle_loop.mp3 (nếu có), khuếch đại + fade-in. Idempotent."""
        if not os.path.exists(self._idle_audio_path):
            return
        if self._idle_proc and self._idle_proc.poll() is None:
            return   # đang phát rồi
        try:
            self._idle_proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                 "-loop", "0",
                 "-af", f"volume={IDLE_VOLUME},afade=t=in:d={IDLE_FADE_IN}",
                 self._idle_audio_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[UI] Không phát được nhạc IDLE: {e}", flush=True)
            self._idle_proc = None

    def _stop_idle_audio(self):
        """Tắt nhạc IDLE gọn gàng — KHÔNG đụng âm lượng PulseAudio (tránh làm
        câm luôn TTS, vì cả hai đều là ffplay và PulseAudio nhớ volume theo app)."""
        proc = self._idle_proc
        self._idle_proc = None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def update_viewfinder(self, frame: np.ndarray):
        with self._lock:
            self.result_img = frame.copy()

    def stop(self):
        self._running = False

    def wait_for_tap(self, poll_interval: float = 0.1) -> str:
        """Block cho đến khi user chạm vào 1 trong 2 thẻ lúc IDLE. Trả về 'kham_da' hoặc 'tu_van'."""
        self.tap_event.clear()
        self.selected_action = None
        while not self.tap_event.is_set():
            if self.crashed or not self._running:
                raise RuntimeError(
                    "Giao diện đã dừng do lỗi render — không thể chờ tap."
                )
            self.tap_event.wait(timeout=poll_interval)
        action = self.selected_action or "kham_da"
        self.tap_event.clear()
        return action

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == State.IDLE:
                # Kiểm tra chạm nút: Quét mã QR / Đăng nhập (dưới câu chào 'Mình là ThinkMed')
                if (30 <= x <= 330 and 140 <= y <= 195):
                    print("[UI] 🖱️ Người dùng nhấn nút: QUÉT MÃ QR", flush=True)
                    self.selected_action = "qr_login"
                    self.tap_event.set()
                # Kiểm tra chạm thẻ 1: Khám da (x: 185..505, y: 380..585)
                elif 185 <= x <= 505 and 380 <= y <= 585:
                    print("[UI] 🖱️ Người dùng nhấn thẻ: KHÁM DA", flush=True)
                    self.selected_action = "kham_da"
                    self.tap_event.set()
                # Kiểm tra chạm thẻ 2: Tư vấn (x: 515..845, y: 380..585)
                elif 515 <= x <= 845 and 380 <= y <= 585:
                    print("[UI] 🖱️ Người dùng nhấn thẻ: TƯ VẤN", flush=True)
                    self.selected_action = "tu_van"
                    self.tap_event.set()
            elif self.state == State.QR_LOGIN:
                # Nút [X Bỏ qua / Thoát] ở góc trên bên phải khi đang mở Quét QR
                if (x >= self.W - 220) and (y <= 75):
                    print("[UI] 🖱️ Người dùng bỏ qua quét mã QR", flush=True)
                    self.exit_event.set()
                    with self._lock:
                        self.state = State.IDLE
            elif self.state == State.VIEWFINDER:
                # Nút [X Kết thúc] ở góc trên bên phải khi đang mở Camera
                if (x >= self.W - 220) and (y <= 75):
                    self.exit_event.set()
                    with self._lock:
                        self.state = State.IDLE
                else:
                    self.capture_event.set()
            elif self.state == State.RESULT_OK:
                # Nút cảm ứng [X Kết thúc] ở góc trên bên phải
                if (x >= self.W - 220) and (y <= 75):
                    self.exit_event.set()
                    try:
                        from audio.tts import stop_audio
                        stop_audio()
                    except Exception:
                        pass
                    with self._lock:
                        self.state = State.IDLE
                # Nút [CÓ, ĐẶT LỊCH] khi đang ở giai đoạn hỏi đặt lịch (540 <= x <= 735, 398 <= y <= 446)
                elif self.result_data.get("booking_stage") == "ask" and (540 <= x <= 735 and 398 <= y <= 446):
                    print("[UI] 🖱️ Người dùng chạm nút: CÓ, ĐẶT LỊCH", flush=True)
                    self.booking_choice = "yes"
                    self.booking_event.set()
                # Nút [KHÔNG, ĐỂ SAU] khi đang ở giai đoạn hỏi đặt lịch (760 <= x <= 955, 398 <= y <= 446)
                elif self.result_data.get("booking_stage") == "ask" and (760 <= x <= 955 and 398 <= y <= 446):
                    print("[UI] 🖱️ Người dùng chạm nút: KHÔNG, ĐỂ SAU", flush=True)
                    self.booking_choice = "no"
                    self.booking_event.set()

            elif self.state in (State.RESULT_MULTI, State.RESULT_NONE,
                                State.LISTENING, State.SPEAKING, State.ANALYZING):
                # Nút cảm ứng [X Kết thúc] ở góc trên bên phải
                if (x >= self.W - 220) and (y <= 75):
                    self.exit_event.set()
                    try:
                        from audio.tts import stop_audio
                        stop_audio()
                    except Exception:
                        pass
                    # Phản hồi tức thì (< 0.01s): Chuyển ngay sang màn hình chờ IDLE
                    with self._lock:
                        self.state = State.IDLE

    # ── Main render loop — PHẢI chạy ở MAIN THREAD ──────────────────────────────

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow(WINDOW_NAME, self.W, self.H)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        try:
            while self._running:
                canvas = np.full((self.H, self.W, 3), BG, dtype=np.uint8)
                self.frame_n += 1

                with self._lock:
                    state = self.state
                    data  = self.result_data.copy()
                    img   = self.result_img.copy() if self.result_img is not None else None

                if   state == State.IDLE:         self._draw_idle(canvas)
                elif state == State.QR_LOGIN:     self._draw_qr_login(canvas, img)
                elif state == State.GREETING:     self._draw_greeting(canvas)
                elif state == State.VIEWFINDER:   self._draw_viewfinder(canvas, img)
                elif state == State.ANALYZING:    self._draw_analyzing(canvas)
                elif state == State.RESULT_OK:    self._draw_result_ok(canvas, img, data)
                elif state == State.RESULT_MULTI: self._draw_result_multi(canvas, img, data)
                elif state == State.RESULT_NONE:  self._draw_result_none(canvas, data)
                elif state == State.LISTENING:    self._draw_listening(canvas)
                elif state == State.SPEAKING:     self._draw_speaking(canvas, data)

                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(33) & 0xFF

                if key == 27:    # ESC
                    self._running = False
                    try:
                        from audio.tts import stop_audio
                        stop_audio()
                    except Exception:
                        pass
                elif key in (ord('q'), ord('Q')):
                    self.exit_event.set()
                    try:
                        from audio.tts import stop_audio
                        stop_audio()
                    except Exception:
                        pass
                elif key == 32:  # Space — chỉ dùng để chụp ảnh nhanh trong VIEWFINDER (không kích hoạt IDLE)
                    if state == State.VIEWFINDER:
                        self.capture_event.set()
                elif key in (ord('1'), ord('2'), ord('3')):
                    from qr_login import QR_USERS
                    key_map = {ord('1'): "ID001", ord('2'): "ID002", ord('3'): "ID003"}
                    uid = key_map.get(key)
                    if uid and uid in QR_USERS:
                        self.logged_in_user = QR_USERS[uid]
                        print(f"\n[UI] ⌨️ Phím tắt nhanh: Đã chọn {self.logged_in_user['name']} ({uid})", flush=True)
                        if state == State.QR_LOGIN:
                            self.set_state(State.QR_LOGIN, qr_user=self.logged_in_user, qr_pts=None, success=True)

        except cv2.error as e:
            print(f"\n[UI] ❌ Lỗi render OpenCV: {e}", flush=True)
            self.crashed  = True
            self._running = False
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── Text helper (tiếng Việt qua PIL) ─────────────────────────────────────

    def get_font(self, size: int, bold: bool = False, italic: bool = False):
        """
        Lấy font TrueType hỗ trợ tiếng Việt đầy đủ cho mọi hệ điều hành (Windows, Linux, Jetson).
        Ưu tiên:
          1. Font trong thư mục assets/fonts/ của dự án (arial.ttf, arialbd.ttf, ariali.ttf).
          2. Font hệ thống Linux phổ biến (/usr/share/fonts/truetype/dejavu, liberation, freefont).
          3. Font hệ thống Windows (C:\\Windows\\Fonts\\arial.ttf).
        """
        key = (size, bold, italic)
        if key in self._font_cache:
            return self._font_cache[key]

        from PIL import ImageFont

        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_name = "arialbd.ttf" if bold else ("ariali.ttf" if italic else "arial.ttf")
        local_font_path = os.path.join(base_dir, "assets", "fonts", font_name)

        candidates = [
            local_font_path,
            # Font Ubuntu / Jetson Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf" if italic else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else (
                "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf" if italic else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ),
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else (
                "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf" if italic else "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
            ),
            # Font Windows
            "C:\\Windows\\Fonts\\arialbd.ttf" if bold else ("C:\\Windows\\Fonts\\ariali.ttf" if italic else "C:\\Windows\\Fonts\\arial.ttf"),
            "C:\\Windows\\Fonts\\segoeuib.ttf" if bold else "C:\\Windows\\Fonts\\segoeui.ttf",
            font_name,
        ]

        loaded_font = None
        for p in candidates:
            if os.path.exists(p):
                try:
                    loaded_font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass

        if loaded_font is None:
            try:
                loaded_font = ImageFont.truetype(font_name, size)
            except Exception:
                try:
                    loaded_font = ImageFont.truetype("DejaVuSans.ttf", size)
                except Exception:
                    loaded_font = ImageFont.load_default()

        self._font_cache[key] = loaded_font
        return loaded_font

    def _put_text(self, canvas, text, pos, size=32, color=WHITE,
                  center=False, bold=False):
        from PIL import Image, ImageDraw

        font = self.get_font(size, bold=bold)

        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = pos
        if center:
            x = x - tw // 2

        rgb_color = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=rgb_color)

        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        canvas[:] = result
        return tw, th

    # ── State renders ─────────────────────────────────────────────────────────

    def _draw_idle(self, c):
        if self._idle_bg is not None:
            t = self.frame_n / 30.0
            blinking = (int(t) % 4 == 0) and (t % 1) < 0.18
            if blinking and self._idle_blink is not None:
                c[:] = self._idle_blink.copy()
            else:
                c[:] = self._idle_bg.copy()

            # 1. Cập nhật đồng hồ thời gian thực ở góc trên bên phải
            cur_time = time.strftime("%H:%M")
            self._put_text(c, cur_time, (895, 34), size=24, color=(100, 50, 20), center=True, bold=True)

            # 2. Hiển thị nút / thông tin Đăng nhập QR ở góc trên bên trái
            from PIL import Image, ImageDraw
            img_pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            btn_font = self.get_font(15, bold=True)

            if self.logged_in_user:
                uname = self.logged_in_user.get("name", "Bệnh nhân")
                uid = self.logged_in_user.get("id", "")
                badge_text = f"BN: {uname} ({uid})"
                draw.rounded_rectangle([36, 145, 320, 190], radius=10, fill=(18, 55, 38), outline=(40, 210, 120), width=2)
                draw.text((48, 156), badge_text, font=btn_font, fill=(190, 255, 210))
            else:
                draw.rounded_rectangle([36, 145, 260, 190], radius=10, fill=(15, 40, 75), outline=(0, 190, 255), width=2)
                draw.text((48, 156), "QUÉT MÃ QR ĐĂNG NHẬP", font=btn_font, fill=(0, 235, 255))

            c[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback nếu thiếu file assets
            cx1, cx2 = self.W // 2 - 220, self.W // 2 + 220
            cy = self.H // 2
            t = self.frame_n / 30.0
            px = int(math.sin(t * 1.6) * 40)
            py = int(math.sin(t * 2.3) * 26)
            blinking = int(t) % 2 == 0 and (t % 1) < 0.12

            for cx in [cx1, cx2]:
                if blinking:
                    cv2.ellipse(c, (cx, cy), (150, 10), 0, 0, 360, WHITE, -1)
                else:
                    cv2.ellipse(c, (cx, cy), (150, 175), 0, 0, 360, WHITE, -1)
                    cv2.circle(c, (cx + px, cy + py), 95, BG, -1)
                    cv2.circle(c, (cx + px + 20, cy + py - 26), 22, WHITE, -1)

            if self.logged_in_user:
                self._put_text(c, f"BN: {self.logged_in_user['name']} ({self.logged_in_user['id']})", (160, 40), size=18, color=GREEN, center=True, bold=True)
            else:
                self._put_text(c, "QUÉT MÃ QR ĐĂNG NHẬP", (140, 40), size=18, color=AMBER, center=True, bold=True)

            self._put_text(c, "ThinkMed", (self.W // 2, 40), size=28,
                           color=BRAND, center=True, bold=True)
            self._put_text(c, "Khám da", (350, 480), size=28, color=WHITE, center=True, bold=True)
            self._put_text(c, "Tư vấn", (674, 480), size=28, color=WHITE, center=True, bold=True)

    def _draw_greeting(self, c):
        self._draw_idle(c)
        action = getattr(self, "selected_action", "kham_da")
        if action == "tu_van":
            banner_text = "Đang kết nối Bác sĩ tư vấn ThinkMed..."
        else:
            banner_text = "Xin chào! Đang chuẩn bị camera khám da..."
        # Banner chào nổi bật phía trên 2 thẻ
        bx1, by1 = self.W // 2 - 280, 320
        bx2, by2 = self.W // 2 + 280, 370
        cv2.rectangle(c, (bx1, by1), (bx2, by2), (240, 120, 0), -1)
        self._put_text(c, banner_text, (self.W // 2, by1 + 12),
                       size=20, color=WHITE, center=True, bold=True)

    def _draw_qr_login(self, c, img):
        """
        Giao diện Quét mã QR nhận diện người dùng:
          - Khung ngắm kiosk bo góc của ui_viewfinder_base.png
          - Camera feed trực tiếp
          - Khung ngắm quét mã QR với 4 góc neon và tia laser chuyển động
          - Thông báo xác nhận đăng nhập thành công với Họ tên + Mã bệnh nhân
          - Nút [X Bỏ qua] ở góc trên bên phải
        """
        qr_user = self.result_data.get("qr_user")
        qr_pts = self.result_data.get("qr_pts")
        is_success = self.result_data.get("success", False)

        if self._viewfinder_base is not None:
            c[:] = self._viewfinder_base.copy()

            vx1, vy1 = 78, 74
            vx2, vy2 = 983, 480
            vw, vh = vx2 - vx1, vy2 - vy1
            r = 36

            vp_mask = np.zeros((vh, vw), dtype=np.uint8)
            cv2.rectangle(vp_mask, (r, 0), (vw - r, vh), 255, -1)
            cv2.rectangle(vp_mask, (0, r), (vw, vh - r), 255, -1)
            cv2.circle(vp_mask, (r, r), r, 255, -1)
            cv2.circle(vp_mask, (vw - r, r), r, 255, -1)
            cv2.circle(vp_mask, (r, vh - r), r, 255, -1)
            cv2.circle(vp_mask, (vw - r, vh - r), r, 255, -1)

            if img is not None:
                resized_cam = cv2.resize(img, (vw, vh))

                # Vẽ viền xanh quanh mã QR nếu camera bắt được pts
                if qr_pts is not None and len(qr_pts) > 0:
                    try:
                        orig_h, orig_w = img.shape[:2]
                        scale_x = vw / orig_w
                        scale_y = vh / orig_h
                        pts_scaled = (qr_pts * [scale_x, scale_y]).astype(int)
                        cv2.polylines(resized_cam, [pts_scaled], True, (0, 255, 120), 4)
                    except Exception:
                        pass

                roi = c[vy1:vy2, vx1:vx2]
                roi[vp_mask > 0] = resized_cam[vp_mask > 0]

            # Khung quét mã QR ở trung tâm (kích thước 240x240)
            cx = (vx1 + vx2) // 2
            cy = (vy1 + vy2) // 2
            box_s = 240
            qx1, qy1 = cx - box_s // 2, cy - box_s // 2
            qx2, qy2 = cx + box_s // 2, cy + box_s // 2

            # 4 góc khung quét neon
            cor_len = 32
            cor_col = (0, 255, 120) if is_success else (0, 220, 255)
            # Top-left
            cv2.line(c, (qx1, qy1), (qx1 + cor_len, qy1), cor_col, 4)
            cv2.line(c, (qx1, qy1), (qx1, qy1 + cor_len), cor_col, 4)
            # Top-right
            cv2.line(c, (qx2, qy1), (qx2 - cor_len, qy1), cor_col, 4)
            cv2.line(c, (qx2, qy1), (qx2, qy1 - cor_len), cor_col, 4)
            # Bottom-left
            cv2.line(c, (qx1, qy2), (qx1 + cor_len, qy2), cor_col, 4)
            cv2.line(c, (qx1, qy2), (qx1, qy2 - cor_len), cor_col, 4)
            # Bottom-right
            cv2.line(c, (qx2, qy2), (qx2 - cor_len, qy2), cor_col, 4)
            cv2.line(c, (qx2, qy2), (qx2, qy2 - cor_len), cor_col, 4)

            # Tia laser quét di chuyển lên xuống nếu chưa đăng nhập
            if not is_success:
                t = self.frame_n / 18.0
                scan_y = int(qy1 + ((math.sin(t) + 1.0) / 2.0) * box_s)
                cv2.line(c, (qx1 + 4, scan_y), (qx2 - 4, scan_y), (0, 235, 255), 2)
                cv2.line(c, (qx1 + 4, scan_y), (qx2 - 4, scan_y), (255, 255, 255), 1)

            # Phủ mascot robot chibi nếu có
            if self._viewfinder_mascot is not None:
                mx1, my1 = 0, 135
                mh, mw = self._viewfinder_mascot.shape[:2]
                m_roi = c[my1:my1+mh, mx1:mx1+mw]
                alpha = self._viewfinder_mascot[:, :, 3].astype(float) / 255.0
                alpha_3 = np.dstack([alpha, alpha, alpha])
                c[my1:my1+mh, mx1:mx1+mw] = (
                    self._viewfinder_mascot[:, :, :3] * alpha_3 + m_roi * (1 - alpha_3)
                ).astype(np.uint8)

            # PIL text render cho tiêu đề và thông báo tiếng Việt
            from PIL import Image, ImageDraw
            img_pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            font_head = self.get_font(22, bold=True)
            font_title = self.get_font(19, bold=True)
            font_card = self.get_font(23, bold=True)
            font_sub = self.get_font(15)
            font_bot = self.get_font(14)

            # 1. Che tiêu đề gốc và vẽ thanh tiêu đề ĐĂNG NHẬP QR chuyên nghiệp
            draw.rounded_rectangle([260, 6, 764, 66], radius=14, fill=(12, 38, 72), outline=(0, 210, 255), width=2)
            head_txt = "QUÉT MÃ QR - ĐĂNG NHẬP"
            hb = draw.textbbox((0, 0), head_txt, font=font_head)
            draw.text((self.W // 2 - (hb[2] - hb[0]) // 2, 22), head_txt, font=font_head, fill=(255, 255, 255))

            # 2. Che phụ đề gốc và thay bằng hướng dẫn quét QR
            draw.rectangle([210, 68, 810, 94], fill=(6, 20, 42))
            guide_txt = "VUI LÒNG HƯỚNG MÃ QR VÀO Ô VUÔNG ĐỂ NHẬN DIỆN"
            gb = draw.textbbox((0, 0), guide_txt, font=font_sub)
            draw.text((self.W // 2 - (gb[2] - gb[0]) // 2, 72), guide_txt, font=font_sub, fill=(0, 230, 255))

            # 3. Che thanh trạng thái dưới chân và thay bằng hướng dẫn
            draw.rounded_rectangle([290, 508, 730, 580], radius=18, fill=(8, 28, 55), outline=(30, 95, 160), width=1)
            b1_txt = "Tự động nhận diện thông tin bệnh nhân qua camera"
            b1_b = draw.textbbox((0, 0), b1_txt, font=font_bot)
            draw.text((self.W // 2 - (b1_b[2] - b1_b[0]) // 2, 520), b1_txt, font=font_bot, fill=(220, 240, 255))
            b2_txt = "Hoặc nhấn nút [X Kết thúc] góc trên để bỏ qua"
            b2_b = draw.textbbox((0, 0), b2_txt, font=font_bot)
            draw.text((self.W // 2 - (b2_b[2] - b2_b[0]) // 2, 546), b2_txt, font=font_bot, fill=(130, 165, 200))

            if is_success and qr_user:
                # Bảng thông báo ĐĂNG NHẬP THÀNH CÔNG
                card_x1, card_y1 = cx - 280, cy - 75
                card_x2, card_y2 = cx + 280, cy + 85
                draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=16,
                                       fill=(10, 55, 35), outline=(0, 255, 120), width=3)
                t_suc = "ĐÃ XÁC THỰC DANH TÍNH THÀNH CÔNG"
                tb = draw.textbbox((0, 0), t_suc, font=font_title)
                draw.text((cx - (tb[2]-tb[0])//2, card_y1 + 16), t_suc, font=font_title, fill=(100, 255, 150))

                t_name = f"Bệnh nhân: {qr_user['name']} ({qr_user['id']})"
                nb = draw.textbbox((0, 0), t_name, font=font_card)
                draw.text((cx - (nb[2]-nb[0])//2, card_y1 + 54), t_name, font=font_card, fill=(255, 255, 255))

                t_wait = "Đang chuyển vào hệ thống..."
                wb = draw.textbbox((0, 0), t_wait, font=font_sub)
                draw.text((cx - (wb[2]-wb[0])//2, card_y1 + 102), t_wait, font=font_sub, fill=(180, 230, 200))

            c[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback nếu thiếu file asset nền
            if img is not None:
                h, w = img.shape[:2]
                scale = min(self.W * 0.7 / w, self.H * 0.8 / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (nw, nh))
                x0 = (self.W - nw) // 2
                y0 = (self.H - nh) // 2
                c[y0:y0+nh, x0:x0+nw] = resized
            self._put_text(c, "HƯỚNG MÃ QR VÀO CAMERA ĐỂ ĐĂNG NHẬP", (self.W // 2, 40),
                           size=22, color=BRAND, center=True, bold=True)
            if is_success and qr_user:
                self._put_text(c, f"ĐÃ NHẬN DIỆN: {qr_user['name']} ({qr_user['id']})",
                               (self.W // 2, self.H - 50), size=22, color=GREEN, center=True, bold=True)

    def _draw_viewfinder(self, c, img):
        locked    = self.result_data.get("locked", False)
        countdown = self.result_data.get("countdown")

        if self._viewfinder_base is not None:
            c[:] = self._viewfinder_base.copy()

            # Tọa độ khung ngắm bên trong viền xanh neon bo góc
            vx1, vy1 = 78, 74
            vx2, vy2 = 983, 480
            vw, vh = vx2 - vx1, vy2 - vy1
            r = 36

            # Mặt nạ bo tròn góc khung ngắm
            vp_mask = np.zeros((vh, vw), dtype=np.uint8)
            cv2.rectangle(vp_mask, (r, 0), (vw - r, vh), 255, -1)
            cv2.rectangle(vp_mask, (0, r), (vw, vh - r), 255, -1)
            cv2.circle(vp_mask, (r, r), r, 255, -1)
            cv2.circle(vp_mask, (vw - r, r), r, 255, -1)
            cv2.circle(vp_mask, (r, vh - r), r, 255, -1)
            cv2.circle(vp_mask, (vw - r, vh - r), r, 255, -1)

            if img is not None:
                resized_cam = cv2.resize(img, (vw, vh))
                roi = c[vy1:vy2, vx1:vx2]
                roi[vp_mask > 0] = resized_cam[vp_mask > 0]

            # Khung định vị vùng da (Dynamic Guide Box)
            gx1, gy1, gx2, gy2 = 136, 119, 916, 461
            gr = 24

            def draw_rounded_rect(target_img, pt1, pt2, color, radius, thickness):
                x1, y1 = pt1
                x2, y2 = pt2
                cv2.line(target_img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
                cv2.line(target_img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
                cv2.line(target_img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
                cv2.line(target_img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
                cv2.ellipse(target_img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
                cv2.ellipse(target_img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
                cv2.ellipse(target_img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
                cv2.ellipse(target_img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

            if locked:
                box_color = (0, 255, 100)   # Xanh lá neon rực rỡ
                glow_color = (0, 180, 60)
            else:
                box_color = (0, 235, 255)   # Vàng tươi
                glow_color = (0, 160, 200)

            draw_rounded_rect(c, (gx1, gy1), (gx2, gy2), glow_color, gr, 6)
            draw_rounded_rect(c, (gx1, gy1), (gx2, gy2), box_color, gr, 3)

            # Phủ linh vật robot chibi lớp nổi trong suốt (RGBA)
            if self._viewfinder_mascot is not None:
                mx1, my1 = 0, 135
                mh, mw = self._viewfinder_mascot.shape[:2]
                m_roi = c[my1:my1+mh, mx1:mx1+mw]
                alpha = self._viewfinder_mascot[:, :, 3].astype(float) / 255.0
                alpha_3 = np.dstack([alpha, alpha, alpha])
                c[my1:my1+mh, mx1:mx1+mw] = (
                    self._viewfinder_mascot[:, :, :3] * alpha_3 + m_roi * (1 - alpha_3)
                ).astype(np.uint8)

            # Cập nhật phụ đề và số đếm ngược thời gian thực khi đã khóa vùng da
            if locked and countdown is not None:
                from PIL import Image, ImageDraw
                img_pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font_sub = self.get_font(16, bold=True)
                font_count = self.get_font(58, bold=True)

                # Thay thế dòng phụ đề bằng dải thông báo xanh lá
                draw.rectangle([250, 56, 770, 78], fill=(6, 20, 42))
                sub_txt = f"ĐÃ KHÓA VÙNG DA! TỰ ĐỘNG CHỤP TRONG {countdown}S (GIỮ YÊN NHÉ)"
                sb = draw.textbbox((0, 0), sub_txt, font=font_sub)
                sw = sb[2] - sb[0]
                draw.text((self.W // 2 - sw // 2, 59), sub_txt, font=font_sub, fill=(100, 255, 120))

                # Vòng tròn số đếm ngược nổi bật giữa khung hình
                ccx, ccy = self.W // 2, (gy1 + gy2) // 2
                cnt_text = str(countdown)
                cb = draw.textbbox((0, 0), cnt_text, font=font_count)
                cw, ch = cb[2] - cb[0], cb[3] - cb[1]
                draw.ellipse([ccx - 52, ccy - 52, ccx + 52, ccy + 52],
                             fill=(0, 190, 80), outline=(255, 255, 255), width=4)
                draw.text((ccx - cw // 2, ccy - ch // 2 - 4), cnt_text, font=font_count, fill=(255, 255, 255))

                c[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback nếu thiếu file asset nền
            if img is not None:
                h, w = img.shape[:2]
                scale = min(self.W * 0.7 / w, self.H * 0.8 / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (nw, nh))
                x0 = (self.W - nw) // 2
                y0 = (self.H - nh) // 2
                c[y0:y0+nh, x0:x0+nw] = resized

                margin = 60
                if locked:
                    cv2.rectangle(c, (x0 + margin, y0 + margin),
                                 (x0 + nw - margin, y0 + nh - margin), (0, 255, 0), 4)
                else:
                    alpha  = abs(math.sin(self.frame_n * 0.1))
                    color  = (int(100 * alpha), 255, int(64 + 191 * alpha))
                    cv2.rectangle(c, (x0 + margin, y0 + margin),
                                 (x0 + nw - margin, y0 + nh - margin), color, 3)

            if locked and countdown:
                self._put_text(c, f"ĐÃ KHÓA VÙNG DA! GIỮ NGUYÊN ({countdown}s)",
                               (self.W // 2, self.H - 45), size=26,
                               color=GREEN, center=True, bold=True)
                self._put_text(c, "Đang chuẩn bị chụp tự động...",
                               (self.W // 2, 40), size=24,
                               color=GREEN, center=True)
            else:
                self._put_text(c, "Đưa vùng da cần kiểm tra vào ô vuông",
                               (self.W // 2, self.H - 45), size=26,
                               color=WHITE, center=True)
                self._put_text(c, "(Chạm màn hình hoặc bấm Space để chụp ngay)",
                               (self.W // 2, 40), size=22,
                               color=GRAY, center=True)
            self._draw_exit_button(c)

    def _draw_analyzing(self, c):
        cx, cy = self.W // 2, self.H // 2 - 60
        for i in range(8):
            angle = (self.frame_n * 6 + i * 45) * math.pi / 180
            x = cx + int(70 * math.cos(angle))
            y = cy + int(70 * math.sin(angle))
            v = int(255 * (i / 8))
            cv2.circle(c, (x, y), 8, (255, v, v), -1)

        self._put_text(c, "Đang phân tích...", (self.W // 2, cy + 100),
                       size=40, color=BLUE, center=True, bold=True)

    def _draw_exit_button(self, c):
        """Nút cảm ứng [X Kết thúc] ở góc trên bên phải dành cho màn hình Kiosk Robot."""
        x1, y1, x2, y2 = self.W - 175, 14, self.W - 18, 62
        # Bóng đổ nhẹ
        cv2.rectangle(c, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (15, 10, 15), -1)
        # Nền màu đỏ đô / ruby cao cấp
        cv2.rectangle(c, (x1, y1), (x2, y2), (35, 30, 175), -1)
        # Viền đỏ phát sáng tương phản cao
        cv2.rectangle(c, (x1, y1), (x2, y2), (70, 70, 240), 2)
        # Vẽ biểu tượng vector dấu X sắc nét bằng cv2.line (chống lỗi font Unicode)
        ix, iy = x1 + 26, (y1 + y2) // 2
        arm = 7
        cv2.line(c, (ix - arm, iy - arm), (ix + arm, iy + arm), WHITE, 2, cv2.LINE_AA)
        cv2.line(c, (ix - arm, iy + arm), (ix + arm, iy - arm), WHITE, 2, cv2.LINE_AA)
        # Text "Kết thúc" chuẩn font tiếng Việt
        self._put_text(c, "Kết thúc", (ix + 16, y1 + 10), size=18, color=WHITE, center=False, bold=True)

    def _put_multiline_text(self, canvas, text, pos, max_width, size=21, color=WHITE, line_spacing=10):
        """Xuống dòng tự nhiên cho văn bản phụ đề trên màn hình."""
        words = text.split()
        lines = []
        cur_line = []
        for w in words:
            test_line = " ".join(cur_line + [w])
            if len(test_line) * (size * 0.58) > max_width and cur_line:
                lines.append(" ".join(cur_line))
                cur_line = [w]
            else:
                cur_line.append(w)
        if cur_line:
            lines.append(" ".join(cur_line))

        x, y = pos
        for line in lines[:8]:
            self._put_text(canvas, line, (x, y), size=size, color=color, center=True)
            y += size + line_spacing

    def _draw_result_ok(self, c, img, data):
        if self._result_bg is not None:
            c[:] = self._result_bg.copy()

            # 1. Vẽ ảnh camera chụp vùng da lên khung bên phải (Viewport bo tròn x: 520..975, y: 168..460)
            if img is not None:
                vx1, vy1, vx2, vy2 = 520, 168, 975, 460
                vw, vh = vx2 - vx1, vy2 - vy1
                r = 20

                vp_mask = np.zeros((vh, vw), dtype=np.uint8)
                cv2.rectangle(vp_mask, (r, 0), (vw - r, vh), 255, -1)
                cv2.rectangle(vp_mask, (0, r), (vw, vh - r), 255, -1)
                cv2.circle(vp_mask, (r, r), r, 255, -1)
                cv2.circle(vp_mask, (vw - r, r), r, 255, -1)
                cv2.circle(vp_mask, (r, vh - r), r, 255, -1)
                cv2.circle(vp_mask, (vw - r, vh - r), r, 255, -1)

                resized_cam = cv2.resize(img, (vw, vh))
                roi = c[vy1:vy2, vx1:vx2]
                roi[vp_mask > 0] = resized_cam[vp_mask > 0]

            # 2. Vẽ thanh tiến trình độ tin cậy AI (Progress Bar Gradient Cyan x: 68..408, y: 326..352)
            confidences = data.get("confidences", [])
            if not confidences and "confidence" in data:
                confidences = [data.get("confidence", 0.0)]
            conf = confidences[0] if confidences else 0.95

            bx1, by1, bx2, by2 = 68, 326, 408, 352
            bw, bh = bx2 - bx1, by2 - by1
            fill_w = max(18, int(bw * max(0.05, min(1.0, conf))))
            br = 12

            bar_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.rectangle(bar_mask, (br, 0), (fill_w - br, bh), 255, -1)
            cv2.rectangle(bar_mask, (0, br), (fill_w, bh - br), 255, -1)
            cv2.circle(bar_mask, (br, br), br, 255, -1)
            cv2.circle(bar_mask, (fill_w - br, br), br, 255, -1)
            cv2.circle(bar_mask, (br, bh - br), br, 255, -1)
            cv2.circle(bar_mask, (fill_w - br, bh - br), br, 255, -1)

            fill_img = np.zeros((bh, bw, 3), dtype=np.uint8)
            for x in range(fill_w):
                ratio = x / bw
                c_b = int(255 * (1 - ratio * 0.1))
                c_g = int(235 * (1 - ratio * 0.15))
                c_r = 0
                fill_img[:, x] = (c_b, c_g, c_r)

            bar_roi = c[by1:by2, bx1:bx2]
            bar_roi[bar_mask > 0] = fill_img[bar_mask > 0]

            # 3. Vẽ văn bản bằng PIL: Tên bệnh tiếng Việt, Tỷ lệ %, Trạng thái giọng nói
            from PIL import Image, ImageDraw
            img_pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            font_name = self.get_font(34, bold=True)
            font_pct = self.get_font(24, bold=True)
            font_status = self.get_font(19, bold=True)
            font_card_tag = self.get_font(11, bold=True)
            font_card_title = self.get_font(15, bold=True)
            font_card_loc = self.get_font(13)
            font_modal_title = self.get_font(18, bold=True)
            font_modal_sub = self.get_font(13)
            font_modal_room = self.get_font(16, bold=True)
            font_modal_loc = self.get_font(14)
            font_modal_hint = self.get_font(13, italic=True)
            font_btn = self.get_font(16, bold=True)
            font_stt_num = self.get_font(58, bold=True)
            font_stt_lbl = self.get_font(13, bold=True)

            # Tên bệnh tiếng Việt (căn giữa trong Pill Box center_x=268, center_y=244)
            vi_name = data.get("vi_name", "Không rõ")
            bb_name = draw.textbbox((0, 0), vi_name, font=font_name)
            nw, nh = bb_name[2] - bb_name[0], bb_name[3] - bb_name[1]
            draw.text((268 - nw // 2, 244 - nh // 2 - 2), vi_name, font=font_name, fill=(20, 60, 160))

            # Phần trăm độ tin cậy (VD: 96%) tại x=455, y=339
            pct_str = f"{conf:.0%}"
            bb_pct = draw.textbbox((0, 0), pct_str, font=font_pct)
            pw, ph = bb_pct[2] - bb_pct[0], bb_pct[3] - bb_pct[1]
            draw.text((455 - pw // 2, 339 - ph // 2 - 2), pct_str, font=font_pct, fill=(20, 60, 160))

            # Gợi ý phòng khám chuyên khoa
            clinic_room = data.get("clinic_room")
            if not clinic_room:
                from config import get_clinic_room
                clinic_room = get_clinic_room(data.get("disease_key"))

            card_x1, card_y1, card_x2, card_y2 = 68, 396, 470, 467
            # Đổ bóng nhẹ
            draw.rounded_rectangle([card_x1 + 1, card_y1 + 1, card_x2 + 1, card_y2 + 1], radius=10, fill=(210, 230, 250))
            # Nền thẻ trắng viền xanh y tế
            draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=10, fill=(255, 255, 255), outline=(50, 130, 220), width=2)
            # Tag pill gợi ý phân luồng hoặc trạng thái da khỏe mạnh
            if data.get("disease_key") == "healthy":
                draw.rounded_rectangle([card_x1 + 8, card_y1 + 6, card_x1 + 242, card_y1 + 21], radius=4, fill=(225, 250, 235))
                draw.text((card_x1 + 14, card_y1 + 7), "TRẠNG THÁI SỨC KHỎE LÀN DA", font=font_card_tag, fill=(10, 130, 60))
                draw.text((card_x1 + 12, card_y1 + 25), "Tình trạng: Da khỏe mạnh bình thường", font=font_card_title, fill=(15, 110, 50))
                draw.text((card_x1 + 12, card_y1 + 46), "Khuyến nghị: Duy trì làm sạch & chống nắng", font=font_card_loc, fill=(50, 95, 70))
            else:
                draw.rounded_rectangle([card_x1 + 8, card_y1 + 6, card_x1 + 242, card_y1 + 21], radius=4, fill=(225, 240, 255))
                draw.text((card_x1 + 14, card_y1 + 7), "GỢI Ý PHÂN LUỒNG KHÁM CHUYÊN KHOA", font=font_card_tag, fill=(10, 80, 175))
                # Số phòng & tên chuyên khoa
                draw.text((card_x1 + 12, card_y1 + 25), f"{clinic_room['room_num']}: {clinic_room['room_name']}", font=font_card_title, fill=(15, 45, 120))
                # Vị trí khu vực phòng khám
                draw.text((card_x1 + 12, card_y1 + 46), f"Vị trí: {clinic_room['location']}", font=font_card_loc, fill=(70, 95, 130))

            # 4. Vẽ bảng Đặt lịch khám / Phiếu khám STT 14 bên khung phải nếu đang trong quy trình đặt lịch
            rx1, ry1, rx2, ry2 = 520, 168, 975, 460
            booking_stage = data.get("booking_stage")
            stt_num = data.get("stt_num", 14)
            patient_name = data.get("patient_name")
            if not patient_name:
                patient_name = self.logged_in_user.get("name", "Bệnh nhân") if self.logged_in_user else "Bệnh nhân"

            if booking_stage == "ask":
                # Thẻ hỏi đặt lịch
                draw.rounded_rectangle([rx1 + 2, ry1 + 2, rx2 + 2, ry2 + 2], radius=16, fill=(200, 220, 245))
                draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=16, fill=(255, 255, 255), outline=(50, 130, 225), width=2)

                draw.rounded_rectangle([rx1, ry1, rx2, ry1 + 48], radius=14, fill=(232, 243, 255))
                draw.rectangle([rx1, ry1 + 32, rx2, ry1 + 48], fill=(232, 243, 255))
                draw.line([(rx1, ry1 + 48), (rx2, ry1 + 48)], fill=(180, 210, 245), width=1)
                draw.text((rx1 + 20, ry1 + 14), "ĐẶT LỊCH KHÁM CHUYÊN KHOA", font=font_modal_title, fill=(15, 65, 150))

                draw.text((rx1 + 22, ry1 + 64), "Hệ thống có thể lấy số thứ tự khám trực tiếp cho bạn tại:", font=font_modal_sub, fill=(60, 80, 110))
                draw.rounded_rectangle([rx1 + 20, ry1 + 90, rx2 - 20, ry1 + 155], radius=10, fill=(245, 249, 255), outline=(100, 160, 240), width=1)
                draw.text((rx1 + 35, ry1 + 102), f"{clinic_room['room_num']}: {clinic_room['room_name']}", font=font_modal_room, fill=(10, 45, 125))
                draw.text((rx1 + 35, ry1 + 128), f"Vị trí: {clinic_room['location']} | Khám chuyên khoa", font=font_modal_loc, fill=(75, 100, 130))

                draw.text((rx1 + 35, ry1 + 168), "Bạn có muốn đặt lịch khám không? (Nói 'Có' / 'Không' hoặc chạm nút)", font=font_modal_hint, fill=(80, 110, 150))

                # Nút [CÓ, ĐẶT LỊCH]
                by1_x1, by1_y1, by1_x2, by1_y2 = rx1 + 20, ry2 - 62, rx1 + 215, ry2 - 14
                draw.rounded_rectangle([by1_x1, by1_y1, by1_x2, by1_y2], radius=10, fill=(16, 160, 90))
                btn_yes_txt = "CÓ, ĐẶT LỊCH"
                by_b = draw.textbbox((0, 0), btn_yes_txt, font=font_btn)
                draw.text((by1_x1 + (by1_x2 - by1_x1 - (by_b[2]-by_b[0]))//2, by1_y1 + 14), btn_yes_txt, font=font_btn, fill=(255, 255, 255))

                # Nút [KHÔNG, ĐỂ SAU]
                by2_x1, by2_y1, by2_x2, by2_y2 = rx2 - 215, ry2 - 62, rx2 - 20, ry2 - 14
                draw.rounded_rectangle([by2_x1, by2_y1, by2_x2, by2_y2], radius=10, fill=(225, 235, 245), outline=(160, 185, 210), width=1)
                btn_no_txt = "KHÔNG, ĐỂ SAU"
                bn_b = draw.textbbox((0, 0), btn_no_txt, font=font_btn)
                draw.text((by2_x1 + (by2_x2 - by2_x1 - (bn_b[2]-bn_b[0]))//2, by2_y1 + 14), btn_no_txt, font=font_btn, fill=(90, 110, 135))

            elif booking_stage == "confirmed":
                # Phiếu khám STT 14
                draw.rounded_rectangle([rx1 + 2, ry1 + 2, rx2 + 2, ry2 + 2], radius=16, fill=(200, 225, 215))
                draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=16, fill=(255, 255, 255), outline=(20, 160, 90), width=2)

                draw.rounded_rectangle([rx1, ry1, rx2, ry1 + 48], radius=14, fill=(230, 248, 238))
                draw.rectangle([rx1, ry1 + 32, rx2, ry1 + 48], fill=(230, 248, 238))
                draw.line([(rx1, ry1 + 48), (rx2, ry1 + 48)], fill=(160, 220, 190), width=1)
                draw.text((rx1 + 20, ry1 + 14), "PHIẾU ĐĂNG KÝ KHÁM BỆNH ĐIỆN TỬ", font=font_modal_title, fill=(10, 120, 65))

                # STT Badge
                stt_box_x1, stt_box_y1, stt_box_x2, stt_box_y2 = rx1 + 20, ry1 + 60, rx1 + 155, ry1 + 195
                draw.rounded_rectangle([stt_box_x1, stt_box_y1, stt_box_x2, stt_box_y2], radius=12, fill=(238, 250, 244), outline=(30, 175, 95), width=2)
                draw.text((stt_box_x1 + 18, stt_box_y1 + 12), "SỐ THỨ TỰ", font=font_stt_lbl, fill=(15, 120, 60))
                stt_str = str(stt_num)
                sb = draw.textbbox((0, 0), stt_str, font=font_stt_num)
                draw.text((stt_box_x1 + (stt_box_x2 - stt_box_x1 - (sb[2]-sb[0]))//2, stt_box_y1 + 40), stt_str, font=font_stt_num, fill=(10, 150, 75))
                draw.text((stt_box_x1 + 32, stt_box_y1 + 108), "(Ưu tiên)", font=font_modal_sub, fill=(50, 150, 90))

                dx = rx1 + 175
                draw.text((dx, ry1 + 64), "Bệnh nhân:", font=font_modal_sub, fill=(100, 115, 130))
                draw.text((dx + 82, ry1 + 63), patient_name, font=font_modal_room, fill=(15, 45, 110))
                draw.text((dx, ry1 + 95), "Phòng khám:", font=font_modal_sub, fill=(100, 115, 130))
                draw.text((dx + 85, ry1 + 94), f"{clinic_room['room_num']}", font=font_modal_room, fill=(10, 140, 70))
                draw.text((dx, ry1 + 122), f"Chuyên khoa: {clinic_room['room_name']}", font=font_modal_sub, fill=(50, 75, 110))
                draw.text((dx, ry1 + 148), f"Vị trí: {clinic_room['location']}", font=font_modal_sub, fill=(50, 75, 110))
                draw.text((dx, ry1 + 174), "Thời gian: 09:30 AM (Ước tính còn 2 người)", font=font_modal_sub, fill=(60, 90, 120))

                draw.rounded_rectangle([rx1 + 15, ry2 - 75, rx2 - 15, ry2 - 15], radius=10, fill=(230, 248, 238), outline=(50, 180, 100), width=1)
                draw.text((rx1 + 35, ry2 - 62), "[ ĐÃ ĐẶT LỊCH THÀNH CÔNG VÀO HỆ THỐNG ]", font=font_modal_room, fill=(10, 130, 60))
                draw.text((rx1 + 35, ry2 - 38), "Xin mời bạn theo bảng chỉ dẫn để di chuyển đến phòng khám.", font=font_modal_sub, fill=(40, 110, 70))

            elif booking_stage == "declined":
                # Banner nhắc nhở nhẹ dưới khung camera
                draw.rounded_rectangle([rx1 + 15, ry2 - 55, rx2 - 15, ry2 - 12], radius=8, fill=(245, 248, 255), outline=(180, 205, 235), width=1)
                draw.text((rx1 + 25, ry2 - 42), "Bạn đã chọn không đặt lịch. Bạn có thể tự đến khám khi thuận tiện.", font=font_modal_sub, fill=(60, 85, 120))

            # Trạng thái robot ở thanh dưới cùng + chấm cyan chớp nháy tại x=214, y=502
            speaking = data.get("speaking", False)
            if speaking:
                dots = "." * ((self.frame_n // 8) % 4)
                status_txt = f"Robot đang đọc kết quả, vui lòng lắng nghe{dots}"
            elif booking_stage == "ask":
                status_txt = f"Bạn có muốn đặt lịch khám tại {clinic_room['room_num']} không?"
            elif booking_stage == "confirmed":
                status_txt = f"Đặt khám thành công! STT: {stt_num} tại {clinic_room['room_num']}"
            elif data.get("disease_key") == "healthy":
                status_txt = "Da khỏe mạnh! Mời bạn đặt câu hỏi chăm sóc da..."
            else:
                status_txt = "Mời bạn đặt câu hỏi hoặc mô tả thêm..."
            draw.text((214, 502), status_txt, font=font_status, fill=(20, 60, 160))

            c[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        else:
            # Fallback nếu thiếu file asset nền
            mid = self.W // 2

            self._put_text(c, "Phát hiện khả năng:", (mid // 2, 80),
                           size=26, color=AMBER, center=True)

            vi_name = data.get("vi_name", "Không rõ")
            confidences = data.get("confidences", [])
            if not confidences and "confidence" in data:
                confidences = [data.get("confidence", 0.0)]

            self._put_text(c, vi_name, (mid // 2, 260), size=38,
                           color=WHITE, center=True, bold=True)

            clinic_room = data.get("clinic_room")
            if not clinic_room:
                from config import get_clinic_room
                clinic_room = get_clinic_room(data.get("disease_key"))

            self._put_text(c, f"Gợi ý: {clinic_room['room_num']} - {clinic_room['room_name']}",
                           (mid // 2, 395), size=18, color=GREEN, center=True, bold=True)
            self._put_text(c, f"Vị trí: {clinic_room['location']}",
                           (mid // 2, 424), size=16, color=WHITE, center=True)

            booking_stage = data.get("booking_stage")
            stt_num = data.get("stt_num", 14)
            if booking_stage == "ask":
                self._put_text(c, f"Đặt lịch khám tại {clinic_room['room_num']}? (Có / Không)",
                               (mid // 2, 455), size=18, color=AMBER, center=True, bold=True)
            elif booking_stage == "confirmed":
                self._put_text(c, f"ĐÃ ĐẶT LỊCH: STT {stt_num} ({clinic_room['room_num']})",
                               (mid // 2, 455), size=20, color=GREEN, center=True, bold=True)

            if len(confidences) == 1:
                conf = confidences[0]
                bar_w, bar_x, bar_y = 300, mid // 2 - 150, 340
                cv2.rectangle(c, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), GRAY, -1)
                cv2.rectangle(c, (bar_x, bar_y),
                             (bar_x + int(bar_w * conf), bar_y + 20), GREEN, -1)
                self._put_text(c, f"{conf:.0%}", (mid // 2, bar_y + 30),
                               size=22, color=WHITE, center=True)
            else:
                n = len(confidences)
                self._put_text(c, f"{n} vị trí phát hiện", (mid // 2, 310),
                               size=22, color=GRAY, center=True)

                bar_w, gap = 88, 14
                total_w  = n * bar_w + (n - 1) * gap
                start_x  = mid // 2 - total_w // 2
                bar_y    = 350

                for i, conf in enumerate(confidences):
                    bx = start_x + i * (bar_w + gap)
                    cv2.rectangle(c, (bx, bar_y), (bx + bar_w, bar_y + 16), GRAY, -1)
                    cv2.rectangle(c, (bx, bar_y),
                                 (bx + int(bar_w * conf), bar_y + 16), GREEN, -1)
                    self._put_text(c, f"{conf:.0%}", (bx + bar_w // 2, bar_y + 22),
                                   size=16, color=WHITE, center=True)

            cv2.line(c, (mid, 40), (mid, self.H - 40), GRAY, 1)

            if img is not None:
                self._blit_img_right(c, img, mid)

            speaking = data.get("speaking", False)
            if speaking:
                self._put_text(c, "Robot đang đọc kết quả, vui lòng lắng nghe...", (20, self.H - 40),
                               size=20, color=GREEN, bold=True)
            else:
                self._put_text(c, "Mời bạn đặt câu hỏi hoặc mô tả thêm...", (20, self.H - 40),
                               size=20, color=BLUE, bold=True)
            self._draw_exit_button(c)

    def _draw_result_multi(self, c, img, data):
        mid = self.W // 2

        self._put_text(c, "Có thể là một trong:", (mid // 2, 70),
                       size=28, color=AMBER, center=True)

        candidates = data.get("candidates", [])
        for i, cand in enumerate(candidates[:2]):
            y0 = 160 + i * 160
            self._put_text(c, cand["vi_name"], (mid // 2, y0), size=34,
                           color=WHITE, center=True, bold=True)
            bar_w, bar_x = 320, mid // 2 - 160
            cv2.rectangle(c, (bar_x, y0 + 60), (bar_x + bar_w, y0 + 80), GRAY, -1)
            color = BLUE if i == 0 else AMBER
            cv2.rectangle(c, (bar_x, y0 + 60),
                         (bar_x + int(bar_w * cand["confidence"]), y0 + 80),
                         color, -1)
            self._put_text(c, f"{cand['confidence']:.0%}",
                           (mid // 2, y0 + 90), size=20, color=WHITE, center=True)

        self._put_text(c, "Hãy mô tả triệu chứng để xác định rõ hơn",
                       (mid // 2, 500), size=26, color=AMBER, center=True)

        cv2.line(c, (mid, 40), (mid, self.H - 40), GRAY, 1)
        if img is not None:
            self._blit_img_right(c, img, mid)
        self._draw_exit_button(c)

    def _draw_result_none(self, c, data=None):
        data = data or {}
        if self._result_none_bg is not None:
            c[:] = self._result_none_bg.copy()
        else:
            # Fallback nếu thiếu file asset nền
            for i, (line, size, color) in enumerate([
                ("Không phát hiện bệnh rõ ràng", 38, WHITE),
                ("Vui lòng mô tả triệu chứng:",  28, GRAY),
                ("ngứa, đau, màu sắc, vị trí...", 28, GRAY),
            ]):
                self._put_text(c, line, (self.W // 2, self.H // 2 - 80 + i * 55),
                               size=size, color=color, center=True)

            speaking = data.get("speaking", False)
            if speaking:
                self._put_text(c, "Robot đang đọc tư vấn, vui lòng lắng nghe...", (self.W // 2, self.H - 50),
                               size=20, color=GREEN, center=True, bold=True)
            else:
                self._put_text(c, "Mời bạn mô tả triệu chứng qua micro...", (self.W // 2, self.H - 50),
                               size=20, color=BLUE, center=True, bold=True)
            self._draw_exit_button(c)

    def _draw_listening(self, c):
        cx, cy = 512, 282
        with self._lock:
            speaking = self.is_user_speaking
            energy = self.audio_energy

        if self._listening_bg is not None:
            c[:] = self._listening_bg.copy()

            # Chuẩn hóa mức âm lượng từ 0.0 (thì thầm/im lặng) đến 1.0 (nói rất to)
            vol_norm = min(1.0, max(0.0, (energy - 0.003) / 0.05))

            n_bars = 9
            spacing = 18
            envelope = [0.25, 0.45, 0.7, 0.95, 1.2, 1.0, 0.8, 0.55, 0.3]
            bar_w = 6

            if speaking:
                title = "Đang thu âm giọng nói..."
                if vol_norm < 0.22:
                    sub = "Đã bắt được giọng nói — bạn đang nói nhỏ (hãy nói to hơn một chút)"
                elif vol_norm < 0.65:
                    sub = "Đã bắt được giọng nói — âm lượng rất tốt"
                else:
                    sub = "Đã bắt được giọng nói — âm lượng to & rõ"

                self._put_text(c, title, (cx, 85), size=32, color=WHITE, center=True, bold=True)
                self._put_text(c, sub, (cx, 132), size=19, color=(255, 235, 210), center=True)

                # Vòng sáng quanh micro nở rộng theo độ lớn giọng nói
                mic_glow_radius = int(68 + 20 * vol_norm)
                cv2.circle(c, (cx, cy), mic_glow_radius, (255, 200, 50), 2, cv2.LINE_AA)

                # Sóng âm giật nảy và co giãn độ cao theo âm lượng nói lớn hay nhỏ
                for i in range(n_bars):
                    env = envelope[i]
                    dyn = abs(math.sin(self.frame_n * 0.4 + i * 0.6))
                    base_h = 15 + 130 * vol_norm
                    h = int(base_h * env * (0.45 + 0.55 * dyn))
                    h = max(8, min(160, h))

                    bx_left = 425 - (n_bars - 1 - i) * spacing
                    bx_right = 600 + (n_bars - 1 - i) * spacing

                    glow_c = (255, int(230 - 30 * vol_norm), int(50 * (1 - vol_norm)))
                    core_c = (255, 255, 240)

                    for bx in (bx_left, bx_right):
                        cv2.line(c, (bx, cy - h // 2), (bx, cy + h // 2), (180, 100, 0), bar_w + 6, cv2.LINE_AA)
                        cv2.line(c, (bx, cy - h // 2), (bx, cy + h // 2), glow_c, bar_w + 2, cv2.LINE_AA)
                        cv2.line(c, (bx, cy - h // 2), (bx, cy + h // 2), core_c, bar_w - 2, cv2.LINE_AA)
            else:
                title = "Đang lắng nghe..."
                sub = "Mời bạn đặt câu hỏi hoặc mô tả triệu chứng..."
                self._put_text(c, title, (cx, 85), size=32, color=WHITE, center=True, bold=True)
                self._put_text(c, sub, (cx, 132), size=19, color=(255, 235, 210), center=True)

                # Sóng âm nhịp thở dịu nhẹ khi chờ nói
                for i in range(n_bars):
                    env = envelope[i]
                    dyn = abs(math.sin(self.frame_n * 0.12 + i * 0.4))
                    h = int((10 + 22 * env) * (0.5 + 0.5 * dyn))

                    bx_left = 425 - (n_bars - 1 - i) * spacing
                    bx_right = 600 + (n_bars - 1 - i) * spacing

                    glow_c = (255, 215, 0)
                    core_c = (255, 255, 230)

                    for bx in (bx_left, bx_right):
                        cv2.line(c, (bx, cy - h // 2), (bx, cy + h // 2), (160, 90, 0), bar_w + 4, cv2.LINE_AA)
                        cv2.line(c, (bx, cy - h // 2), (bx, cy + h // 2), glow_c, bar_w + 1, cv2.LINE_AA)
                        cv2.line(c, (bx, cy - h // 2), (bx, cy + h // 2), core_c, bar_w - 3, cv2.LINE_AA)
        else:
            # Fallback nếu chưa có ảnh nền
            if speaking:
                title = "Đang thu âm giọng nói..."
                color = (40, 130, 255)
                self._put_text(c, title, (cx, cy - 140), size=38,
                               color=color, center=True, bold=True)
                n = 22
                energy_factor = min(1.0, max(0.2, energy / 0.02))
                for i in range(n):
                    dyn = math.sin(self.frame_n * 0.3 + i * 0.4)
                    h = int(25 + 130 * energy_factor * abs(dyn))
                    x = cx - n * 14 + i * 28
                    cv2.rectangle(c, (x, cy - h // 2), (x + 16, cy + h // 2), color, -1)
                self._put_text(c, "Đã bắt được giọng nói — hãy tiếp tục nói...", (cx, cy + 120),
                               size=22, color=color, center=True)
            else:
                title = "Đang lắng nghe..."
                color = BLUE
                self._put_text(c, title, (cx, cy - 140), size=38,
                               color=color, center=True, bold=True)
                n = 22
                for i in range(n):
                    h = int(16 + 25 * abs(math.sin(self.frame_n * 0.12 + i * 0.4)))
                    x = cx - n * 14 + i * 28
                    cv2.rectangle(c, (x, cy - h // 2), (x + 16, cy + h // 2), color, -1)
                self._put_text(c, "Mời bạn nói triệu chứng hoặc câu hỏi vào micro", (cx, cy + 120),
                               size=22, color=GRAY, center=True)
            self._draw_exit_button(c)

    def _draw_speaking(self, c, data=None):
        data = data or {}
        text = data.get("text", "")

        if self._speaking_bg is not None:
            c[:] = self._speaking_bg.copy()

            if text:
                from PIL import Image, ImageDraw
                img_pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font = self.get_font(22)

                lines = []
                max_w = 760
                paragraphs = text.split("\n")
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    words = paragraph.split()
                    cur_line = []
                    for w in words:
                        test_line = " ".join(cur_line + [w])
                        bb = draw.textbbox((0, 0), test_line, font=font)
                        tw = bb[2] - bb[0]
                        if tw > max_w and cur_line:
                            lines.append(" ".join(cur_line))
                            cur_line = [w]
                        else:
                            cur_line.append(w)
                    if cur_line:
                        lines.append(" ".join(cur_line))

                x, y = 158, 178
                line_h = 32
                for line in lines[:9]:
                    draw.text((x, y), line, font=font, fill=(255, 255, 255))
                    y += line_h

                c[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback nếu thiếu file asset nền
            cx = self.W // 2
            self._put_text(c, "Bác sĩ AI đang trả lời...", (cx, 28), size=26,
                           color=WHITE, center=True, bold=True)
            if text:
                box_x1, box_y1 = 60, 115
                box_x2, box_y2 = self.W - 60, self.H - 50
                cv2.rectangle(c, (box_x1, box_y1), (box_x2, box_y2), (32, 22, 22), -1)
                cv2.rectangle(c, (box_x1, box_y1), (box_x2, box_y2), (70, 50, 50), 2)
                self._put_multiline_text(c, text, (cx, box_y1 + 25), max_width=self.W - 160,
                                         size=21, color=WHITE, line_spacing=12)
            self._draw_exit_button(c)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _blit_img_right(self, c, img, mid):
        h, w = img.shape[:2]
        max_w, max_h = mid - 60, self.H - 80
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        x0 = mid + (mid - nw) // 2
        y0 = (self.H - nh) // 2
        c[y0:y0+nh, x0:x0+nw] = resized
