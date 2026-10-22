#!/usr/bin/env python3
"""
test_groq_model.py — Kiểm tra model Groq mới trả về gì (chẩn đoán câu cụt).

Chạy:  export GROQ_API_KEY=...
       python test_groq_model.py
"""
import os
from groq import Groq

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

print(f"Model: {MODEL}\n")

# ── Test 1: KHÔNG stream, xem toàn bộ response ──
print("="*50)
print("TEST 1: non-streaming (xem cấu trúc đầy đủ)")
print("="*50)
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role":"user","content":"Nguyên nhân bệnh vảy nến là gì? Trả lời 3 câu."}],
    max_tokens=300,
    temperature=0.3,
)
msg = resp.choices[0].message
print(f"content: {repr(msg.content)}")
# reasoning model có thể có trường reasoning
if hasattr(msg, "reasoning") and msg.reasoning:
    print(f"\n⚠ CÓ trường 'reasoning' (dài {len(msg.reasoning)} ký tự) — đây có thể là chỗ nội dung bị lạc")
print(f"\nfinish_reason: {resp.choices[0].finish_reason}")

# ── Test 2: streaming, đếm token ──
print("\n" + "="*50)
print("TEST 2: streaming (đếm delta.content)")
print("="*50)
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role":"user","content":"Nguyên nhân bệnh vảy nến là gì? Trả lời 3 câu."}],
    max_tokens=300,
    temperature=0.3,
    stream=True,
)
n_chunks = 0
full = ""
for chunk in stream:
    tok = chunk.choices[0].delta.content or ""
    full += tok
    if tok:
        n_chunks += 1
print(f"Số chunk có content: {n_chunks}")
print(f"Tổng text: {repr(full)}")
print(f"Số từ: {len(full.split())}")
