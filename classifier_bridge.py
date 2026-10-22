"""
classifier_bridge.py — Cầu nối SkinClassifier (ảnh) -> pipeline RAG (main.py).

Thay cho run_yolo() cũ. Trả về một ChanDoanResult mô tả ĐẦY ĐỦ tình huống,
để main.py chọn đúng lời thoại thay vì chỉ "có bệnh / không bệnh".

BỐN TÌNH HUỐNG từ classifier.detect():
  reason="ok"             -> có chẩn đoán đủ tự tin  -> mở RAG về bệnh đó
  reason="other"          -> bất thường NGOÀI 5 bệnh -> KHÔNG chẩn đoán, khuyên đi khám
  reason="low_confidence" -> mơ hồ trong 5 bệnh      -> KHÔNG chẩn đoán, khuyên đi khám
  reason="blurry"/"too_*" -> ảnh không đạt           -> yêu cầu chụp lại

NGUYÊN TẮC AN TOÀN (áp cho MỌI nhánh có chẩn đoán):
  Kiosk là công cụ SÀNG LỌC, không phải chẩn đoán. Kể cả khi tự tin, lời thoại
  LUÔN kết thúc bằng khuyến cáo đi khám nếu tổn thương tiến triển. Đây là lớp
  phòng thủ cuối cho các ca ác tính mà ngưỡng confidence không bắt hết.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import (
    CLASSIFIER_MODEL_PATH, DISEASE_VI_NAMES, CONVERSATION_ONLY_DISEASES,
)


@dataclass
class ChanDoanResult:
    reason:       str                       # ok | other | low_confidence | blurry | too_dark | too_bright | no_model
    disease_key:  Optional[str] = None      # chỉ có khi reason="ok"
    confidence:   float = 0.0
    top3:         List[Tuple[str, str, float]] = field(default_factory=list)  # [(key, vi, prob)]
    opening:      str = ""                   # lời mở đầu đã soạn sẵn cho tình huống

    @property
    def has_diagnosis(self) -> bool:
        return self.reason == "ok" and self.disease_key is not None


# ── Lời thoại cho từng tình huống ─────────────────────────────────────

_ASK_RETRY = {
    "blurry":     "Ảnh chưa đủ nét. Bạn giữ máy ổn định và đưa vùng da lại gần khung hơn, rồi chụp lại giúp tôi nhé.",
    "too_dark":   "Ánh sáng hơi tối. Bạn thử di chuyển tới nơi sáng hơn rồi chụp lại nhé.",
    "too_bright": "Ánh sáng hơi chói. Bạn thử tránh ánh sáng trực tiếp rồi chụp lại nhé.",
}

# Câu an toàn — GẮN vào cuối mọi lời thoại có chẩn đoán.
SAFETY_TAIL = "Lưu ý: đây chỉ là công cụ hỗ trợ sàng lọc, không thay thế bác sĩ."

# Khi model KHÔNG chắc (other / low_confidence): không nêu bệnh, hướng về khám.
_UNCERTAIN = (
    "Tôi chưa đủ chắc chắn để đưa ra nhận định về vùng da này. "
    "Bạn có thể mô tả thêm triệu chứng — cảm giác ngứa/đau/rát, thời gian xuất hiện, vị trí — "
    "để tôi hỗ trợ, hoặc tốt nhất là đến gặp bác sĩ da liễu để được khám trực tiếp."
)

# Khi model thấy bất thường rõ nhưng ngoài 5 bệnh (other): nhấn mạnh đi khám hơn.
_OUT_OF_SCOPE = (
    "Vùng da này có dấu hiệu bất thường, nhưng không thuộc nhóm bệnh tôi được huấn luyện để nhận diện. "
    "Bạn nên đến gặp bác sĩ da liễu để được đánh giá đầy đủ. "
    "Trong lúc chờ, bạn có thể mô tả triệu chứng để tôi cung cấp thêm thông tin tham khảo."
)


class ClassifierBridge:
    def __init__(self):
        from classifier import SkinClassifier
        self.clf = SkinClassifier()

    def run(self, image_path: str) -> ChanDoanResult:
        res = self.clf.detect(image_path)

        # ── Ảnh không đạt chất lượng ──
        if res.reason in _ASK_RETRY:
            return ChanDoanResult(reason=res.reason, opening=_ASK_RETRY[res.reason])

        # ── Bất thường ngoài 5 bệnh ──
        if res.reason == "other":
            return ChanDoanResult(reason="other", opening=_OUT_OF_SCOPE,
                                  top3=res.top3 if hasattr(res, "top3") else [])

        # ── Mơ hồ, không đủ tự tin ──
        if res.reason == "low_confidence":
            return ChanDoanResult(reason="low_confidence", opening=_UNCERTAIN,
                                  top3=res.top3 if hasattr(res, "top3") else [])

        # ── Có chẩn đoán đủ tự tin ──
        key = res.best.disease_key
        vi  = DISEASE_VI_NAMES.get(key, key)
        conf = res.best.confidence

        # Soạn opening: nêu bệnh + top-3 + câu an toàn
        lines = [f"Hình ảnh gợi ý khả năng {vi} (độ tin cậy {conf:.0%})."]
        if len(res.top3) > 1:
            alt = ", ".join(f"{v} {p:.0%}" for k, v, p in res.top3[1:3])
            lines.append(f"Một số khả năng khác cần cân nhắc: {alt}.")
        lines.append("Bạn muốn biết gì về tình trạng này? (nguyên nhân, cách chăm sóc, điều trị...)")
        lines.append(SAFETY_TAIL)

        return ChanDoanResult(
            reason="ok", disease_key=key, confidence=conf,
            top3=res.top3, opening=" ".join(lines),
        )
