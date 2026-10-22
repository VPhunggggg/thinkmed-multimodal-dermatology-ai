"""
fuzzy_match.py — Sửa lỗi phát âm/STT khi nhận diện tên bệnh trong câu nói.

Vấn đề: STT (faster-whisper) có thể nhận sai do phát âm không chuẩn,
nhiễu mic, hoặc giới hạn model — vd "vảy nến" → "vây nên", "trị bệnh"
→ "chi bênh". _detect_disease_in_text() cũ dùng exact substring match
nên các case này bị bỏ sót hoàn toàn → query rewrite không có disease_key
để filter RAG, phải dựa hết vào semantic search (may rủi hơn).

Giải pháp 2 lớp:
  1. Bỏ dấu (unidecode) trước khi so sánh — xử lý > 90% lỗi STT tiếng Việt
     vì phần lớn lỗi là lệch/mất dấu, giữ nguyên phụ âm-nguyên âm gốc.
  2. Fuzzy partial_ratio (rapidfuzz) với ngưỡng 80% — bắt thêm lỗi phụ âm
     nhẹ (ch/tr, d/gi...) mà bỏ dấu chưa xử lý hết.

Cài: pip install rapidfuzz unidecode
"""

from typing import Optional

from rapidfuzz import fuzz
from unidecode import unidecode

from config import DISEASE_ALIASES

FUZZY_THRESHOLD = 80   # partial_ratio tối thiểu để coi là khớp


def _norm(s: str) -> str:
    """Bỏ dấu + lowercase — chuẩn hoá để so sánh khoan dung lỗi STT."""
    return unidecode(s.lower().strip())


def find_disease_fuzzy(text: str) -> Optional[str]:
    """
    Tìm tên bệnh trong text, khoan dung lỗi phát âm/STT.

    Thử theo thứ tự (nhanh → chậm):
      1. Exact substring match (giữ hành vi cũ, nhanh nhất — đa số câu
         STT nhận đúng vẫn nên đi đường nhanh, không cần fuzzy)
      2. Substring match sau khi bỏ dấu (bắt lỗi dấu — phổ biến nhất)
      3. Fuzzy partial_ratio sau khi bỏ dấu (bắt lỗi phụ âm nhẹ)

    Trả về disease_key nếu tìm được, None nếu không khớp gì đủ tin cậy.
    """
    lower = text.lower()

    # ── Bước 1: exact match (giữ nguyên hành vi cũ) ──────────────────────────
    for disease_key, aliases in DISEASE_ALIASES.items():
        if any(a in lower for a in aliases):
            return disease_key

    # ── Bước 2: match sau khi bỏ dấu ─────────────────────────────────────────
    norm_text = _norm(text)
    for disease_key, aliases in DISEASE_ALIASES.items():
        for alias in aliases:
            if _norm(alias) in norm_text:
                return disease_key

    # ── Bước 3: fuzzy partial_ratio ───────────────────────────────────────────
    best_key   : Optional[str] = None
    best_score = 0
    for disease_key, aliases in DISEASE_ALIASES.items():
        for alias in aliases:
            score = fuzz.partial_ratio(_norm(alias), norm_text)
            if score > best_score:
                best_score = score
                best_key   = disease_key

    if best_score >= FUZZY_THRESHOLD:
        return best_key

    return None


def debug_match(text: str) -> dict:
    """Trả về chi tiết matching — dùng để test/debug ngưỡng."""
    norm_text = _norm(text)
    results = []
    for disease_key, aliases in DISEASE_ALIASES.items():
        for alias in aliases:
            score = fuzz.partial_ratio(_norm(alias), norm_text)
            results.append((disease_key, alias, score))
    results.sort(key=lambda x: -x[2])
    return {
        "input": text,
        "normalized": norm_text,
        "top_matches": results[:5],
        "result": find_disease_fuzzy(text),
    }


if __name__ == "__main__":
    tests = [
        "tôi bị vây nên",
        "chi bênh vây nên như nào",
        "toi bi vay nen a",
        "bệnh phong có nguy hiểm không",
        "mun trung ca chua the nao",
    ]
    for t in tests:
        r = debug_match(t)
        print(f"'{t}' → {r['result']}  (top: {r['top_matches'][0]})")
