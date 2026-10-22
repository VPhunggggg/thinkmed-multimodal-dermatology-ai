#!/usr/bin/env python3
"""
test_generate.py — Test thẳng generate_response, KHÔNG mở giao diện.

Chẩn đoán lỗi "câu trả lời 2 từ" mà không cần chạy cả robot.
Chạy:
    cd ~/Desktop/skin_disease_rag
    export GROQ_API_KEY=...
    export GROQ_MODEL=openai/gpt-oss-120b
    python test_generate.py
"""
import os
from rag.retriever import Retriever
from rag.response_generator import generate_response

# Lấy chunk thật từ RAG cho câu hỏi vảy nến
r = Retriever()
q = "nguyên nhân bệnh vảy nến"
chunks = r.query(q, disease_key="psoriasis")
print(f"RAG trả về {len(chunks)} chunks\n")

print("="*60)
print("Gọi generate_response (voice_mode=False, không TTS)...")
print("="*60)
resp = generate_response(
    user_query=q,
    retrieved_chunks=chunks,
    disease_key="psoriasis",
    yolo_confidence=0.9,
    history=[],
    voice_mode=False,   # KHÔNG TTS -> cô lập vòng đọc chunk
)
print("\n" + "="*60)
print(f"KẾT QUẢ: {len(resp.split())} từ / {len(resp)} ký tự")
print(f"Nội dung: {repr(resp)}")
print("="*60)

if len(resp.split()) < 10:
    print("\n❌ VẪN CỤT → lỗi ở vòng đọc chunk / reasoning model (không phải TTS)")
    print("   → Áp bản sửa: reasoning_effort='low' + xử lý chunk an toàn")
else:
    print("\n✅ ĐỦ → vòng đọc chunk OK. Nếu robot vẫn cụt thì lỗi ở TTS/voice_mode")
