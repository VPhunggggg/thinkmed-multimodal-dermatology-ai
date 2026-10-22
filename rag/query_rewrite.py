"""
Query Rewriter — history-aware version.

Thay đổi so với bản cũ:
  - build_search_query nhận thêm tham số `history`
  - _expand_followup: phát hiện câu hỏi follow-up ("còn gì nữa?", "cụ thể hơn?")
    và kết hợp với câu hỏi trước để tạo query đầy đủ hơn
"""

from typing import Dict, List, Optional

from config import DISEASE_VI_NAMES


# ── Intent detection ──────────────────────────────────────────────────────────

INTENT_MAP: dict[str, list[str]] = {
    "definition":    ["là gì", "định nghĩa", "khái niệm"],
    "symptoms":      ["triệu chứng", "biểu hiện", "dấu hiệu", "ngứa", "phát ban", "đau"],
    "treatment":     ["điều trị", "chữa", "thuốc", "phương pháp", "trị"],
    "cause":         ["nguyên nhân", "tại sao", "do đâu", "gây ra"],
    "diagnosis":     ["chẩn đoán", "xét nghiệm", "phân biệt"],
    "prevention":    ["phòng ngừa", "phòng tránh", "tránh"],
    "prognosis":     ["tiên lượng", "khỏi không", "bao lâu"],
    "who_at_risk":   ["ai dễ mắc", "đối tượng", "nguy cơ"],
    "complications": ["biến chứng", "nguy hiểm", "nghiêm trọng"],
    "general":       ["thông tin", "tổng quan", "bệnh của tôi", "bị gì"],
}

INTENT_TEMPLATES: dict[str, str] = {
    "definition":    "{vi} là gì định nghĩa bệnh da",
    "symptoms":      "triệu chứng biểu hiện lâm sàng {vi}",
    "treatment":     "điều trị thuốc phương pháp chữa trị {vi}",
    "cause":         "nguyên nhân gây ra {vi} yếu tố nguy cơ",
    "diagnosis":     "chẩn đoán xét nghiệm phân biệt {vi}",
    "prevention":    "phòng ngừa {vi}",
    "prognosis":     "kết quả điều trị tiên lượng {vi}",
    "who_at_risk":   "ai dễ mắc đối tượng nguy cơ {vi}",
    "complications": "biến chứng nguy hiểm {vi}",
    "general":       "{vi} {query}",
}

# Từ khoá nhận biết câu follow-up ngắn thiếu context
FOLLOWUP_KEYWORDS = [
    "còn", "thêm", "nữa", "cụ thể", "ví dụ", "nghĩa là", "sao vậy",
    "tiếp theo", "và", "giải thích", "rõ hơn", "hơn không",
]


def detect_intent(user_query: str) -> str:
    lower = user_query.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "general"


# ── Follow-up expansion ───────────────────────────────────────────────────────

def _expand_followup(query: str, history: Optional[List[Dict]]) -> str:
    """
    Nếu câu hỏi hiện tại ngắn và mang tính follow-up
    ("còn gì nữa?", "cụ thể hơn?", "ví dụ?")
    → kết hợp với câu hỏi user trước để tạo query đầy đủ.

    Ví dụ:
      history[-2] = "triệu chứng bệnh vảy nến là gì"
      query       = "còn gì nữa không"
      → expanded  = "triệu chứng bệnh vảy nến là gì còn gì nữa không"
    """
    if not history:
        return query

    lower = query.lower()
    words = query.split()
    is_short    = len(words) <= 6
    is_followup = any(kw in lower for kw in FOLLOWUP_KEYWORDS)

    if not (is_short and is_followup):
        return query

    # Tìm câu hỏi user gần nhất trong history
    for msg in reversed(history):
        if msg["role"] == "user":
            prev = msg["content"]
            # Chỉ kết hợp nếu câu trước có nội dung rõ ràng
            if len(prev.split()) > 3:
                return f"{prev} {query}"
            break

    return query


# ── Main public function ──────────────────────────────────────────────────────

def build_search_query(
    user_query:  str,
    disease_key: Optional[str] = None,
    use_llm:     bool = False,
    history:     Optional[List[Dict]] = None,
) -> str:
    """
    Tạo search query tối ưu.

    Args:
        user_query:  Câu hỏi hiện tại của user
        disease_key: Bệnh từ YOLO hoặc context hội thoại
        use_llm:     True → dùng LLM rewrite
        history:     Lịch sử hội thoại gần nhất
    """
    vi_name = DISEASE_VI_NAMES.get(disease_key, "") if disease_key else ""

    # Mở rộng follow-up query
    effective_query = _expand_followup(user_query, history)

    intent = detect_intent(effective_query)

    if use_llm:
        return _llm_rewrite(effective_query, vi_name, disease_key, intent, history)
    else:
        return _rule_based_rewrite(effective_query, vi_name, intent)


# ── Rule-based ────────────────────────────────────────────────────────────────

def _rule_based_rewrite(query: str, vi_name: str, intent: str) -> str:
    # Không có tên bệnh → giữ nguyên query, không expand template sai
    if not vi_name.strip():
        return query
    template = INTENT_TEMPLATES.get(intent, "{vi} {query}")
    return template.format(vi=vi_name, query=query).strip()


# ── LLM rewrite ───────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "Bạn là hệ thống viết lại câu truy vấn cho cơ sở dữ liệu y tế. "
    "Chỉ trả về câu truy vấn tối ưu, không giải thích, tối đa 25 từ."
)

def _llm_rewrite(
    query:       str,
    vi_name:     str,
    disease_key: Optional[str],
    intent:      str,
    history:     Optional[List[Dict]],
) -> str:
    """Dùng Groq (khớp LLM_BACKEND) viết lại query. Lỗi → lùi rule-based."""
    from config import GROQ_API_KEY, GROQ_MODEL
    try:
        from groq import Groq
    except ImportError:
        return _rule_based_rewrite(query, vi_name, intent)

    # Tóm tắt context từ history
    history_ctx = ""
    if history:
        last_turns = [m for m in history[-4:] if m["role"] == "user"]
        if last_turns:
            history_ctx = f"Câu hỏi trước: {last_turns[-1]['content']}\n"

    prompt = (
        f"Bệnh: {vi_name or 'không rõ'}\n"
        f"{history_ctx}"
        f"Câu hỏi hiện tại (có thể bị nhận nhầm do giọng nói): {query}\n"
        f"Ý định: {intent}\n\n"
        f"Viết lại thành câu tìm kiếm tiếng Việt ngắn gọn, có tên bệnh và từ khoá "
        f"y tế. Sửa lỗi chính tả/nghe nhầm nếu có."
    )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 60,
            temperature = 0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or _rule_based_rewrite(query, vi_name, intent)
    except Exception as e:
        print(f"[QueryRewrite] LLM lỗi ({e}) — dùng rule-based", flush=True)
        return _rule_based_rewrite(query, vi_name, intent)
