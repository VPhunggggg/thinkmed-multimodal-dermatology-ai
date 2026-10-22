"""
classifier.py — Bộ phân loại bệnh da (drop-in thay cho detector.py)

Thay đổi so với YOLODetector:
  - Không còn detection / bounding box thật. Người dùng tự đưa vùng da vào
    khung ngắm trên UI; module này crop đúng vùng đó rồi phân loại.
  - Có lớp "other" (OOD) → model được phép nói "không biết".
  - Có quality gate (mờ / tối / cháy sáng) chặn trước khi phân loại.
  - Có temperature scaling → xác suất hiển thị cho bệnh nhân đã hiệu chỉnh.
  - Trả về top-k để tầng LLM làm chẩn đoán phân biệt.

Interface giữ nguyên: .detect(image_path) -> kết quả có .best / .disease_key
/ .confidence / .all_detections. `best is None` giữ đúng ngữ nghĩa cũ
("không kết luận được") nên main.py không cần đổi luồng.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_base_dir = Path(__file__).resolve().parent
_ultralytics_dir = str(_base_dir / "ultralytics_custom")
if _ultralytics_dir not in sys.path:
    sys.path.insert(0, _ultralytics_dir)
from ultralytics import YOLO

from config import (
    CLASSIFIER_MODEL_PATH,   # Path tới checkpoint .pt
    CLASS_NAMES,             # list[str] — PHẢI đúng thứ tự lúc train
    CLASS_TO_DISEASE,        # class_name -> disease_key dùng cho RAG
    DISEASE_VI_NAMES,        # disease_key -> tên tiếng Việt
    OTHER_CLASS,             # tên lớp OOD, vd "other"
    CONF_THRESHOLD,          # float, HOẶC dict {class_name: ngưỡng} — xem detect()
    CROP_BOX,                # None, hoặc (x1,y1,x2,y2) toạ độ chuẩn hoá 0..1
    BLUR_MIN,                # Laplacian variance tối thiểu
    BRIGHTNESS_RANGE,        # (min, max) độ sáng trung bình 0..255
    TTA_HFLIP,               # bool
)


# ---------------------------------------------------------------- dataclasses

@dataclass
class Prediction:
    class_name:  str
    disease_key: str
    disease_vi:  str
    confidence:  float
    bbox:        List[float]   # vùng crop [x1, y1, x2, y2] pixel

    @property
    def yolo_class(self) -> str:
        """Alias tương thích ngược với code cũ."""
        return self.class_name


@dataclass
class ClassificationResult:
    best:           Optional[Prediction]
    all_detections: List[Prediction] = field(default_factory=list)
    image_path:     str = ""
    reason:         str = ""   # "ok" | "blurry" | "too_dark" | "too_bright"
                               # | "low_confidence" | "other"
    probs:          Dict[str, float] = field(default_factory=dict)

    @property
    def disease_key(self) -> Optional[str]:
        return self.best.disease_key if self.best else None

    @property
    def confidence(self) -> float:
        return self.best.confidence if self.best else 0.0

    @property
    def top3(self) -> List[Tuple[str, str, float]]:
        """[(disease_key, disease_vi, prob), ...] — dùng để build prompt LLM."""
        return [
            (p.disease_key, p.disease_vi, p.confidence)
            for p in self.all_detections[:3]
        ]


# ---------------------------------------------------------------- classifier

class SkinClassifier:
    """Phân loại bệnh da từ ảnh trong khung ngắm bằng mô hình YOLO11s (task classify)."""

    def __init__(self, model_path: Path = CLASSIFIER_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy classifier checkpoint: {model_path}"
            )

        print(f"[CLS] Đang load model YOLO11s: {model_path}")
        self.model = YOLO(str(model_path))
        self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]

        print(
            f"[CLS] YOLO11s Classes: {self.class_names}"
        )
        self._warmup()

    # ------------------------------------------------------------ internals

    def _warmup(self, n: int = 1) -> None:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, verbose=False)
        print("[CLS] Warmup xong.")

    @staticmethod
    def _crop(bgr: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Cắt theo khung ngắm. CROP_BOX=None nghĩa là ảnh đã được crop sẵn."""
        h, w = bgr.shape[:2]
        if CROP_BOX is None:
            return bgr, [0.0, 0.0, float(w), float(h)]
        x1 = int(CROP_BOX[0] * w)
        y1 = int(CROP_BOX[1] * h)
        x2 = int(CROP_BOX[2] * w)
        y2 = int(CROP_BOX[3] * h)
        return bgr[y1:y2, x1:x2], [float(x1), float(y1), float(x2), float(y2)]

    @staticmethod
    def _quality_check(bgr: np.ndarray) -> Optional[str]:
        """Trả về mã lỗi nếu ảnh không dùng được, None nếu OK."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())
        lo, hi = BRIGHTNESS_RANGE

        if var < BLUR_MIN:
            print(f"[CLS] ⚠️ Ảnh mờ: độ nét var={var:.1f} < ngưỡng {BLUR_MIN}", flush=True)
            return "blurry"

        if mean < lo:
            print(f"[CLS] ⚠️ Ảnh tối: độ sáng mean={mean:.1f} < {lo}", flush=True)
            return "too_dark"
        if mean > hi:
            print(f"[CLS] ⚠️ Ảnh chói sáng: độ sáng mean={mean:.1f} > {hi}", flush=True)
            return "too_bright"

        return None

    # ------------------------------------------------------------ public API

    def detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Phân loại ảnh đã chụp. Giữ nguyên chữ ký của YOLODetector.detect().

        `best is None` xảy ra khi: ảnh không đạt chất lượng, model chọn lớp
        "other", hoặc confidence dưới ngưỡng. Đọc `.reason` để biết vì sao.
        """
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

        crop_bgr, bbox = self._crop(bgr)

        bad = self._quality_check(crop_bgr)
        if bad:
            print(f"[CLS] Ảnh không đạt ({bad}): {image_path}")
            return ClassificationResult(
                best=None, image_path=str(image_path), reason=bad
            )

        # Suy luận trực tiếp qua YOLO11s classify
        yolo_res = self.model.predict(crop_bgr, verbose=False)[0]
        probs = yolo_res.probs.data.tolist()
        prob_map = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        # Xếp hạng, bỏ lớp OOD khỏi danh sách chẩn đoán phân biệt
        ranked = [
            Prediction(
                class_name  = c,
                disease_key = CLASS_TO_DISEASE.get(c, c),
                disease_vi  = DISEASE_VI_NAMES.get(CLASS_TO_DISEASE.get(c, c), c),
                confidence  = prob_map[c],
                bbox        = bbox,
            )
            for c in sorted(self.class_names, key=lambda c: -prob_map[c])
            if c != OTHER_CLASS
        ]

        top_class = max(prob_map, key=prob_map.get)
        top_prob = prob_map[top_class]

        # OOD: model cho rằng đây không thuộc các bệnh nó được huấn luyện
        if top_class == OTHER_CLASS:
            print(f"[CLS] Ngoài phạm vi (other={top_prob:.2f}): {image_path}")
            return ClassificationResult(
                best=None, all_detections=ranked, image_path=str(image_path),
                reason="other", probs=prob_map,
            )

        # Ngưỡng RIÊNG cho từng lớp.
        # `wart`/`acne` chịu ngưỡng cao hơn: giá trị chẩn đoán thấp (mụn cóc
        # thường tự khỏi, trứng cá là thẩm mỹ), nhưng thiệt hại khi sai rất cao
        # — đó là hai chẩn đoán mang nghĩa "lành tính, đừng lo", nên gán nhầm
        # chúng cho một tổn thương ác tính là kịch bản tệ nhất của hệ thống.
        if conf_threshold is not None:
            thr = conf_threshold
        elif isinstance(CONF_THRESHOLD, dict):
            thr = CONF_THRESHOLD.get(top_class, 0.75)
        else:
            thr = CONF_THRESHOLD

        # Không đủ tự tin
        if top_prob < thr:
            print(f"[CLS] Confidence thấp ({top_class}={top_prob:.2f} < {thr})")
            return ClassificationResult(
                best=None, all_detections=ranked, image_path=str(image_path),
                reason="low_confidence", probs=prob_map,
            )

        best = ranked[0]
        print(
            f"[CLS] {best.class_name} → {best.disease_vi} "
            f"(p={best.confidence:.2f})"
        )
        return ClassificationResult(
            best=best, all_detections=ranked, image_path=str(image_path),
            reason="ok", probs=prob_map,
        )


# ------------------------------------------------- alias tương thích ngược
# Nhờ những dòng này, `from detector import YOLODetector` cũ vẫn chạy nếu bạn
# đổi tên file này thành detector.py. Nên xoá sau khi đã dọn main.py.
Detection = Prediction
DetectionResult = ClassificationResult
YOLODetector = SkinClassifier
