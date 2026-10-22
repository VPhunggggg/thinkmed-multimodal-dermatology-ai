"""
YOLO detector wrapper cho model bệnh da đã train.

Mapping YOLO class name → disease_key trong RAG:
  Chỉnh sửa YOLO_TO_DISEASE trong config.py cho khớp tên class model của bạn.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import YOLO_MODEL_PATH, YOLO_TO_DISEASE, DISEASE_VI_NAMES


@dataclass
class Detection:
    yolo_class:  str    # Tên class raw từ YOLO
    disease_key: str    # Canonical key (dùng cho RAG filter)
    disease_vi:  str    # Tên tiếng Việt
    confidence:  float  # YOLO confidence score
    bbox:        List[float]  # [x1, y1, x2, y2] pixel


@dataclass
class DetectionResult:
    best:           Optional[Detection]   # Detection có confidence cao nhất
    all_detections: List[Detection] = field(default_factory=list)
    image_path:     str = ""

    @property
    def disease_key(self) -> Optional[str]:
        return self.best.disease_key if self.best else None

    @property
    def confidence(self) -> float:
        return self.best.confidence if self.best else 0.0


class YOLODetector:
    """Wrapper cho ultralytics YOLO model."""

    def __init__(self, model_path: Path = YOLO_MODEL_PATH):
        from ultralytics import YOLO
        print(f"[YOLO] Đang load model: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy YOLO model: {model_path}\n"
                f"Copy file .pt vào: {model_path}"
            )
        self.model = YOLO(str(model_path))
        print(f"[YOLO] Model classes: {list(self.model.names.values())}")

    def detect(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
    ) -> DetectionResult:
        """
        Chạy YOLO inference trên ảnh JPG/PNG.

        Args:
            image_path:     Đường dẫn tới file ảnh
            conf_threshold: Ngưỡng confidence tối thiểu

        Returns:
            DetectionResult với detection tốt nhất và toàn bộ detections
        """
        results = self.model(
            image_path,
            conf=conf_threshold,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            for box in result.boxes:
                cls_id    = int(box.cls[0])
                cls_name  = result.names[cls_id]
                conf      = float(box.conf[0])
                bbox      = box.xyxy[0].tolist()

                # Map YOLO class → canonical disease key
                disease_key = YOLO_TO_DISEASE.get(cls_name.lower(), cls_name.lower())
                disease_vi  = DISEASE_VI_NAMES.get(disease_key, cls_name)

                detections.append(Detection(
                    yolo_class  = cls_name,
                    disease_key = disease_key,
                    disease_vi  = disease_vi,
                    confidence  = conf,
                    bbox        = bbox,
                ))

        if not detections:
            print(f"[YOLO] Không phát hiện bệnh nào trong: {image_path}")
            return DetectionResult(best=None, image_path=image_path)

        # Chọn detection có confidence cao nhất
        best = max(detections, key=lambda d: d.confidence)
        print(
            f"[YOLO] Phát hiện: {best.yolo_class} → {best.disease_vi} "
            f"(conf={best.confidence:.2f})"
        )

        return DetectionResult(
            best           = best,
            all_detections = detections,
            image_path     = image_path,
        )