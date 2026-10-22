"""
test_pipeline_headless.py — GIAI ĐOẠN 3: kiểm tra bộ não hệ thống.

Chạy pipeline lõi HOÀN TOÀN headless (không camera, không mic, không loa,
không UI), giả lập YOLO đã detect sẵn 1 bệnh:

    user text → build_search_query → Retriever → generate_response → text

Mục đích: chứng minh RAG + LLM + hội thoại nhiều lượt (history) hoạt động
đúng, và đo độ trễ từng khâu — tách khỏi mọi rắc rối I/O.

Cách chạy trên Jetson:
    cd ~/Desktop/skin_disease_rag
    export GROQ_API_KEY="gsk_..."        # nếu chưa export trong ~/.bashrc
    python3 test_pipeline_headless.py
"""

import os
import sys
import time

# ── 0. Kiểm tra điều kiện trước, fail nhanh với thông báo rõ ──────────────────
import config

backend = getattr(config, "LLM_BACKEND", "").lower()
print(f"[Setup] LLM_BACKEND = {backend!r}")

if backend == "groq" and not os.getenv("GROQ_API_KEY"):
    sys.exit(
        "❌ Chưa có GROQ_API_KEY.\n"
        "   Chạy:  export GROQ_API_KEY=\"gsk_...\"  rồi thử lại.\n"
        "   (Hoặc test không cần mạng:  LLM_BACKEND=none python3 test_pipeline_headless.py)"
    )

try:
    if backend == "groq":
        import groq  # noqa: F401
except ImportError:
    sys.exit("❌ Thiếu thư viện groq.  Chạy:  pip install groq")

# ── 1. Nạp RAG (đây là bước chậm nhất: sentence-transformers + chroma) ─────────
from rag.retriever import Retriever
from rag.query_rewrite import build_search_query
from rag.response_generator import generate_response

print("\n[Setup] Đang nạp Retriever (embedding model + ChromaDB)...")
t0 = time.time()
retriever = Retriever()
print(f"[Setup] ✅ Retriever sẵn sàng sau {time.time() - t0:.1f}s\n")

# ── 2. Giả lập YOLO đã detect 1 bệnh có trong RAG ─────────────────────────────
#     (vảy nến — KHÔNG nằm trong DISEASES_WITHOUT_RAG={"wart"})
DISEASE_KEY = "psoriasis"
YOLO_CONF   = 0.91
vi_name     = config.DISEASE_VI_NAMES.get(DISEASE_KEY, DISEASE_KEY)
print(f"[Sim]  Giả lập YOLO phát hiện: {vi_name} ({DISEASE_KEY}), conf={YOLO_CONF:.0%}")
print("=" * 70)

# ── 3. Hội thoại nhiều lượt — mỗi lượt dồn vào history để test câu hỏi nối tiếp ─
TURNS = [
    "bệnh này là gì",           # định nghĩa
    "triệu chứng ra sao",       # follow-up: phải hiểu 'triệu chứng của VẢY NẾN'
    "có chữa khỏi được không",  # follow-up: tiên lượng
]

history: list[dict] = []

for i, user_input in enumerate(TURNS, 1):
    print(f"\n┌─ Lượt {i} ──────────────────────────────────────────────────────")
    print(f"│ 👤 User: {user_input}")

    # 3a. Rewrite query (history-aware)
    t = time.time()
    search_q = build_search_query(user_input, DISEASE_KEY, history=history)
    t_rewrite = (time.time() - t) * 1000
    print(f"│ 🔍 Search query: {search_q!r}   ({t_rewrite:.0f}ms)")

    # 3b. Retrieve
    t = time.time()
    chunks = retriever.query(search_q, disease_key=DISEASE_KEY)
    t_retrieve = (time.time() - t) * 1000
    top = ", ".join(f"{c['section_title']}({c['score']:.2f})" for c in chunks[:3])
    print(f"│ 📚 Lấy {len(chunks)} chunks: {top}   ({t_retrieve:.0f}ms)")

    # 3c. Generate (voice_mode=False → chỉ trả text, KHÔNG đọc loa)
    t = time.time()
    answer = generate_response(
        user_query       = user_input,
        retrieved_chunks = chunks,
        disease_key      = DISEASE_KEY,
        yolo_confidence  = YOLO_CONF,
        history          = history,
        voice_mode       = False,
    )
    t_llm = (time.time() - t) * 1000
    print(f"│ 🤖 LLM ({t_llm:.0f}ms):")
    for line in answer.splitlines():
        print(f"│    {line}")

    # 3d. Cập nhật history cho lượt sau
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})

    print(f"└─ tổng lượt: {t_rewrite + t_retrieve + t_llm:.0f}ms "
          f"(rewrite {t_rewrite:.0f} + retrieve {t_retrieve:.0f} + llm {t_llm:.0f})")

print("\n" + "=" * 70)
print("✅ GIAI ĐOẠN 3 PASS: text-in → RAG → LLM → text-out chạy qua 3 lượt hội thoại.")
print("   Nếu lượt 2/3 trả lời đúng ngữ cảnh vảy nến (không hỏi lại 'bệnh nào?')")
print("   → history-aware hoạt động đúng.")
