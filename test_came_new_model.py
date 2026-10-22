#!/usr/bin/env python3
"""
test_camera.py — Kiểm thử classifier bằng camera CSI thật trên Jetson.

Chạy TRƯỚC khi nối RAG/LLM. Mục tiêu: xác nhận toàn bộ chuỗi
   camera -> khung ngắm -> quality gate -> classifier -> ngưỡng
hoạt động trên phần cứng thật, với ánh sáng thật, da thật.

PHÍM:
   SPACE  chụp khung trong khung ngắm -> phân loại
   c      hiệu chỉnh: in Laplacian variance của khung hiện tại
          (dùng để đặt BLUR_MIN trong config — xem hướng dẫn cuối file)
   t      bật/tắt overlay top-3 xác suất
   q      thoát

Trên Orin Nano, EfficientNet-B0 PyTorch thuần đủ nhanh cho ảnh tĩnh —
không cần TensorRT ở bước này. Nếu sau này latency thành vấn đề, mới
export ONNX -> build .engine TRÊN Jetson (không build trên PC).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# classifier.py + config.py phải nằm cùng thư mục (hoặc trong PYTHONPATH)
from classifier import SkinClassifier
from config_classifier import CROP_BOX, BLUR_MIN, BRIGHTNESS_RANGE


# ── GStreamer pipeline cho camera CSI trên Jetson ──────────────────────
def csi_pipeline(sensor_id=0, w=1280, h=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )


def draw_frame_guide(frame):
    """Vẽ khung ngắm cố định. Người dùng đưa tổn thương vào đây."""
    h, w = frame.shape[:2]
    if CROP_BOX is None:
        return frame, (0, 0, w, h)
    x1, y1 = int(CROP_BOX[0] * w), int(CROP_BOX[1] * h)
    x2, y2 = int(CROP_BOX[2] * w), int(CROP_BOX[3] * h)
    # làm mờ vùng ngoài khung để mắt tập trung vào giữa
    mask = frame.copy()
    cv2.rectangle(mask, (0, 0), (w, h), (0, 0, 0), -1)
    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Dua vung da can kiem tra vao khung",
                (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, (x1, y1, x2, y2)


def crop_of(frame):
    h, w = frame.shape[:2]
    if CROP_BOX is None:
        return frame
    x1, y1 = int(CROP_BOX[0] * w), int(CROP_BOX[1] * h)
    x2, y2 = int(CROP_BOX[2] * w), int(CROP_BOX[3] * h)
    return frame[y1:y2, x1:x2]


REASON_VI = {
    "blurry":         "Anh mo — giu may on dinh, chup lai",
    "too_dark":       "Qua toi — them anh sang",
    "too_bright":     "Qua sang — bot anh sang / bot loa",
    "low_confidence": "Chua du chac chan -> nen di kham bac si",
    "other":          "Bat thuong ngoai 5 benh -> nen di kham bac si",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="csi",
                    help="'csi', hoặc index webcam (vd '0'), hoặc đường dẫn video")
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--flip", type=int, default=0, help="flip-method 0..3")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    print("Đang tải model...")
    clf = SkinClassifier()

    if args.source == "csi":
        # Dùng CSICamera (subprocess gst-launch + ISP preset da liễu) — CÙNG
        # pipeline mà run_kiosk.py dùng, nên BLUR_MIN đo ở đây khớp thực tế.
        from csi_camera import CSICamera
        cap = CSICamera(sensor_id=args.sensor_id, width=args.width, height=args.height)
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise SystemExit(
            "Không mở được camera.\n"
            "  - Nếu vừa Ctrl+C lần trước: sudo systemctl restart nvargus-daemon\n"
            "  - Kiểm tra: v4l2-ctl --list-devices"
        )
    print("[CSI] Camera đang khởi động (CSICamera warmup ~1.5s)...\n")

    Path("captures").mkdir(exist_ok=True)
    show_top3 = True
    last = None
    lap_samples = []

    print("\nSPACE=chụp  c=đo độ nét  t=top3  q=thoát\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue        # CSICamera warmup: frame đầu None
        if args.flip == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.flip == 1:
            frame = cv2.flip(frame, 1)

        disp, _ = draw_frame_guide(frame.copy())

        # Overlay kết quả lần chụp gần nhất
        if last:
            y0 = 30
            for line, col in last:
                cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, col, 2)
                y0 += 26

        cv2.imshow("ThinkMed — test classifier", disp)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        elif k == ord("t"):
            show_top3 = not show_top3

        elif k == ord("c"):
            # Hiệu chỉnh BLUR_MIN: đo độ nét khung trong khung ngắm
            g = cv2.cvtColor(crop_of(frame), cv2.COLOR_BGR2GRAY)
            v = cv2.Laplacian(g, cv2.CV_64F).var()
            b = float(g.mean())
            lap_samples.append(v)
            print(f"  Laplacian={v:8.1f}   brightness={b:6.1f}   "
                  f"(BLUR_MIN hiện={BLUR_MIN}, BRIGHTNESS={BRIGHTNESS_RANGE})")

        elif k == ord(" "):
            # Chụp -> lưu -> chạy đúng pipeline classifier.detect()
            path = f"captures/cap_{int(time.time())}.jpg"
            cv2.imwrite(path, frame)
            t0 = time.time()
            res = clf.detect(path)
            dt = (time.time() - t0) * 1000

            last = []
            if res.best:
                last.append((f"{res.best.disease_vi}  {res.best.confidence:.0%}",
                             (0, 255, 0)))
                last.append(("Cong cu sang loc — khong thay bac si", (0, 200, 200)))
            else:
                last.append((REASON_VI.get(res.reason, res.reason), (0, 165, 255)))

            if show_top3 and res.all_detections:
                for p in res.all_detections[:3]:
                    last.append((f"   {p.disease_vi}: {p.confidence:.0%}",
                                 (200, 200, 200)))
            last.append((f"[{dt:.0f} ms]", (150, 150, 150)))

            print(f"\n{path}  ({dt:.0f} ms)  reason={res.reason}")
            for c, pr in sorted(res.probs.items(), key=lambda x: -x[1]):
                print(f"   {c:20s} {pr:.3f}")

    cap.release()
    cv2.destroyAllWindows()

    if lap_samples:
        arr = np.array(lap_samples)
        print(f"\n── Hiệu chỉnh độ nét ({len(arr)} mẫu) ──")
        print(f"   min={arr.min():.0f}  p25={np.percentile(arr,25):.0f}  "
              f"median={np.median(arr):.0f}  max={arr.max():.0f}")
        print("   Đặt BLUR_MIN ở khoảng giữa nhóm ảnh NÉT và nhóm ảnh MỜ bạn đã đo.")


if __name__ == "__main__":
    main()
