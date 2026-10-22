"""
csi_camera.py — Đọc frame từ CSI camera (IMX219) qua gst-launch-1.0 subprocess.

Tại sao không dùng cv2.VideoCapture trực tiếp:
  - cv2.VideoCapture(CAMERA_ID) đọc raw Bayer chưa debayer qua V4L2
    (đã xác nhận: R=0 G=154 B=0 ở độ phân giải gốc cảm biến 3280x2464).
  - cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER) cần OpenCV build có
    GStreamer support. Bản pip "opencv-python" KHÔNG có (GStreamer: NO).
    Bản system cv2 (/usr/lib/python3/dist-packages) CÓ GStreamer nhưng
    Python cache module theo tên "cv2" — không thể load cả 2 bản cv2
    trong cùng 1 process, nên dù sys.path trỏ đúng, import thứ 2 vẫn
    trả về bản đã cache (pip, không GStreamer).

Giải pháp: chạy gst-launch-1.0 như 1 subprocess riêng biệt, đọc raw BGR
bytes qua stdout pipe, tự ghép thành numpy array bằng tay. Hoàn toàn
không phụ thuộc cv2 GStreamer hay GStreamer Python bindings (gi/PyGObject)
— chỉ cần CLI tool gst-launch-1.0 (có sẵn trên mọi JetPack/L4T).

Có background thread liên tục đọc frame mới nhất vào buffer (drop frame
cũ) — tránh hiện tượng viewfinder bị trễ dần khi vòng lặp tiêu thụ
(YOLO + sleep) chạy chậm hơn tốc độ camera sinh frame (30fps).

API tương thích cv2.VideoCapture: .read() -> (ret, frame), .release()
"""

import atexit
import os
import signal
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np


DEFAULT_SKIN_ISP = {
    # Preset mặc định cho macro da liễu — giữ khá bảo thủ vì không thể tự
    # xem ảnh thật để tinh chỉnh chính xác; nên coi đây là điểm khởi đầu,
    # dùng test_camera_quality.py để so sánh trực quan và điều chỉnh nếu cần.
    "wbmode":       1,     # auto white-balance — an toàn nhất khi chưa
                           # biết chính xác điều kiện ánh sáng thực tế
    "saturation":   1.0,   # giữ màu tự nhiên — không nên tăng vì màu đỏ/
                           # thâm của tổn thương da là dữ liệu chẩn đoán,
                           # bóp méo màu có thể gây hiểu sai
    "tnr-mode":     2,     # 2 = high-quality noise reduction
    "tnr-strength": 0.3,   # nhẹ — giảm nhiễu nhưng không xoá mất texture
                           # da (texture là chi tiết quan trọng cho classifier
                           # phân biệt các loại tổn thương)
    "ee-mode":      2,     # 2 = high-quality edge enhancement
    "ee-strength":  0.3,   # nhẹ — làm rõ viền tổn thương, tránh sharpen
                           # quá tay tạo artifact giả
}


def restart_nvargus_daemon() -> bool:
    """
    Dọn dẹp tiến trình camera bị kẹt và khởi động lại nvargus-daemon trên Jetson.
    Tránh tình trạng phải reboot cả máy khi camera session bị crash hoặc lock.
    """
    print("[CSICamera] 🔄 Đang tự động dọn dẹp và reset nvargus-daemon...", flush=True)
    # 1. Dọn dẹp gst-launch cũ nếu có
    try:
        subprocess.run(["pkill", "-9", "-f", "gst-launch-1.0"], capture_output=True, timeout=3)
    except Exception:
        pass

    # 2. Khởi động lại nvargus-daemon
    success = False
    try:
        res = subprocess.run(
            ["sudo", "systemctl", "restart", "nvargus-daemon"],
            capture_output=True, text=True, timeout=10
        )
        if res.returncode == 0:
            print("[CSICamera] ✅ Đã restart nvargus-daemon thành công.", flush=True)
            success = True
        else:
            print(f"[CSICamera] ⚠️ sudo systemctl restart nvargus-daemon trả về mã {res.returncode}: {res.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"[CSICamera] ⚠️ Không thể gọi systemctl restart nvargus-daemon: {e}", flush=True)

    if not success:
        # Fallback: nếu sudo systemctl không chạy được, thử kill daemon để systemd tự respawn
        try:
            subprocess.run(["sudo", "pkill", "-9", "nvargus-daemon"], capture_output=True, timeout=3)
            success = True
        except Exception:
            pass

    time.sleep(2.0)   # Rất quan trọng: cho daemon thời gian khởi tạo lại socket IPC
    return success


class CSICamera:
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_sec: float = 1.5,
        isp_settings: Optional[dict] = None,
        auto_recover: bool = True,
    ):
        """
        Args:
            isp_settings: Tham số ISP của nvarguscamerasrc (wbmode, saturation,
                tnr-mode, tnr-strength, ee-mode, ee-strength, exposuretimerange...).
                None → dùng DEFAULT_SKIN_ISP. Truyền {} để tắt hoàn toàn.
            auto_recover: True -> tự động restart nvargus-daemon và thử lại nếu
                phát hiện camera session bị kẹt từ phiên trước.
        """
        self.sensor_id    = sensor_id
        self.width        = width
        self.height       = height
        self.fps          = fps
        self.warmup_sec   = warmup_sec
        self.isp_settings = isp_settings
        self.frame_size   = width * height * 3   # BGR 8-bit, không nén

        self._lock          = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stderr_tail   = []
        self._running       = False
        self._released      = False
        self._proc: Optional[subprocess.Popen] = None

        # Khởi động pipeline lần 1
        self._start_pipeline()

        # Nếu mở không thành công (thường do nvargus-daemon bị lock từ lần trước),
        # tự động phục hồi và thử lại nếu bật auto_recover
        if auto_recover and not self.isOpened():
            print("[CSICamera] ⚠️ Pipeline thoát ngay lập tức (nghi vấn nvargus-daemon bị khóa từ phiên trước).", flush=True)
            self._cleanup_proc()
            restart_nvargus_daemon()
            print("[CSICamera] 🔄 Đang thử khởi động lại CSICamera lần 2...", flush=True)
            self._start_pipeline()

        # Đăng ký tự động release khi Python thoát để giải phóng sensor
        atexit.register(self.release)

    def _start_pipeline(self):
        isp = DEFAULT_SKIN_ISP if self.isp_settings is None else self.isp_settings

        pipeline_args = ["nvarguscamerasrc", f"sensor-id={self.sensor_id}"]
        for key, val in isp.items():
            pipeline_args.append(f"{key}={val}")
        pipeline_args += [
            "!",
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},"
            f"framerate={self.fps}/1", "!",
            "nvvidconv", "flip-method=2", "!",
            "video/x-raw,format=BGRx", "!",
            "videoconvert", "!",
            "video/x-raw,format=BGR", "!",
            "fdsink", "fd=1",
        ]

        self._stderr_tail = []
        self._running = True
        self._released = False

        self._proc = subprocess.Popen(
            ["gst-launch-1.0", "-q"] + pipeline_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # Chờ warmup: nếu process chết sớm thì dừng chờ ngay
        t0 = time.time()
        while time.time() - t0 < self.warmup_sec:
            if self._proc.poll() is not None:
                break
            with self._lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.1)

    def _cleanup_proc(self):
        self._running = False
        proc = self._proc
        self._proc = None
        if proc and proc.poll() is None:
            # GStreamer gst-launch cần SIGINT để gửi EOS và ngắt kết nối sạch với nvargus-daemon
            try:
                if hasattr(signal, "SIGINT"):
                    proc.send_signal(signal.SIGINT)
                else:
                    proc.terminate()
            except Exception:
                pass

            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    # ── Background readers ───────────────────────────────────────────────────

    def _drain_stderr(self):
        if not self._proc or not self._proc.stderr:
            return
        try:
            for line in self._proc.stderr:
                text = line.decode(errors="replace").strip()
                if text:
                    self._stderr_tail.append(text)
                    self._stderr_tail = self._stderr_tail[-20:]
        except Exception:
            pass

    def _reader_loop(self):
        while self._running and self._proc:
            raw = self._read_exact(self.frame_size)
            if raw is None:
                break   # pipe đóng — process đã thoát
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self._lock:
                self._latest_frame = frame

    def _read_exact(self, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            if not self._running or self._proc is None or self._proc.stdout is None:
                return None
            try:
                chunk = self._proc.stdout.read(n - len(buf))
                if not chunk:
                    return None   # EOF — process đã thoát
                buf.extend(chunk)
            except Exception:
                return None
        return bytes(buf)

    # ── Public API (giống cv2.VideoCapture) ─────────────────────────────────

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get_stderr_tail(self) -> str:
        """Vài dòng lỗi gần nhất từ gst-launch-1.0 — dùng để debug khi lỗi."""
        return "\n".join(self._stderr_tail)

    def release(self):
        if self._released:
            return
        self._released = True
        self._cleanup_proc()
