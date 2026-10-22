"""
Response Generator — streaming TTS version.

Thay đổi chính:
  _generate_groq() giờ stream từng câu sang TTS queue ngay khi LLM sinh ra,
  thay vì chờ toàn bộ response xong mới bắt đầu đọc.

Pipeline:
  LLM token stream → sentence buffer → TTS queue → audio playback
                                                  (background thread)
"""

import os
import queue
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional

from config import (
    LLM_BACKEND,
    GROQ_API_KEY, GROQ_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OLLAMA_MODEL,
    DISEASE_VI_NAMES, DISEASES_WITHOUT_RAG,
)

SYSTEM_PROMPT = """\
Bạn là trợ lý y tế chuyên về bệnh da liễu, hỗ trợ bệnh nhân trong hội thoại nhiều lượt.

Quy tắc QUAN TRỌNG:
1. Đi thẳng vào nội dung — KHÔNG mở đầu bằng "Cảm ơn", "Dựa trên tài liệu", hay câu xã giao.
2. Nếu tài liệu y tế trong phần TÀI LIỆU THAM KHẢO của system message quá dài bạn tóm lại giúp tôi.
3. Dựa HOÀN TOÀN vào tài liệu y tế trong phần TÀI LIỆU THAM KHẢO của system message.
4. Đọc kỹ LỊCH SỬ HỘI THOẠI — KHÔNG lặp lại thông tin đã nói ở lượt trước.
5. Nếu user hỏi "còn gì nữa" → chỉ bổ sung điểm CHƯA đề cập.
6. Không chẩn đoán xác định, không kê đơn thuốc cụ thể.
7. KHÔNG kết thúc bằng "Còn gì nữa?" — user sẽ tự hỏi tiếp.
8. Chỉ khuyến khích gặp bác sĩ ở câu trả lời quan trọng, không phải mọi lượt.\
"""


# ── Sentence streaming ────────────────────────────────────────────────────────

# Ký tự kết thúc câu
_SENTENCE_END = re.compile(r'(?<=[.!?:])\s+|(?<=\n)\s*')
_MIN_SENTENCE  = 15   # ký tự tối thiểu để TTS đọc (tránh đọc "Ừ.", "OK.")


def _split_sentences(text: str) -> list[str]:
    """Tách text thành các câu để TTS đọc từng câu."""
    # Tách trên dấu . ! ? : và xuống dòng
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁĂÂĐÊÔƠƯÉÍÓÚ])|(?<=\n)', text)
    result = []
    buf = ""
    for p in parts:
        buf += p
        if len(buf) >= _MIN_SENTENCE:
            result.append(buf.strip())
            buf = ""
    if buf.strip():
        result.append(buf.strip())
    return [s for s in result if s]


class _TTSQueue:
    """
    2-stage pipeline để loại bỏ khoảng dừng giữa các câu:

      Stage 1 — gen_thread:  text → MP3 file (edge-tts API call)
      Stage 2 — play_thread: MP3 file → phát qua loa (ffplay)

    Trong khi play_thread đang phát câu N,
    gen_thread đã chuẩn bị xong audio câu N+1 → không còn khoảng dừng.
    """

    _END = object()   # sentinel để báo hiệu kết thúc

    def __init__(self, voice_mode: bool):
        self.voice_mode = voice_mode
        self._text_q  = queue.Queue()   # text đợi generate
        self._audio_q = queue.Queue()   # MP3 path đợi phát

        if voice_mode:
            self._gen_t  = threading.Thread(target=self._gen_worker,  daemon=True)
            self._play_t = threading.Thread(target=self._play_worker, daemon=True)
            self._gen_t.start()
            self._play_t.start()

    def push(self, sentence: str):
        """Đẩy câu vào pipeline."""
        if sentence.strip() and self.voice_mode:
            self._text_q.put(sentence.strip())

    def finish(self):
        """Báo hiệu hết câu và chờ phát xong hoàn toàn."""
        if self.voice_mode:
            self._text_q.put(self._END)
            self._gen_t.join()
            self._play_t.join()

    # ── Stage 1: text → MP3 ──────────────────────────────────────────────────

    def _gen_worker(self):
        import asyncio, tempfile, os
        VOICE = "vi-VN-NamMinhNeural"

        async def _to_mp3(text: str, path: str):
            import edge_tts
            await edge_tts.Communicate(text, VOICE).save(path)

        while True:
            item = self._text_q.get()
            if item is self._END:
                self._audio_q.put(self._END)   # forward sentinel
                break
            try:
                tmp  = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                path = tmp.name
                tmp.close()
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_to_mp3(item, path))
                finally:
                    loop.close()
                self._audio_q.put(path)
            except Exception as e:
                print(f"\n[TTS gen] Lỗi: {e}", flush=True)

    # ── Stage 2: MP3 → loa ───────────────────────────────────────────────────

    def _play_worker(self):
        import os, subprocess
        from audio.tts import _pad_silence
        PLAYERS = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
            ["mpg123", "-q"],
        ]
        first = True
        while True:
            item = self._audio_q.get()
            if item is self._END:
                break
            # Chỉ chèn im lặng cho câu ĐẦU (link BT còn nguội); các câu sau phát
            # liền mạch nên không cần — tránh ngắt quãng giữa các câu.
            play = _pad_silence(item) if first else item
            first = False
            try:
                for cmd in PLAYERS:
                    try:
                        subprocess.run(cmd + [play], check=True, capture_output=True)
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
            except Exception as e:
                print(f"\n[TTS play] Lỗi: {e}", flush=True)
            finally:
                for _p in {item, play}:
                    try:
                        os.unlink(_p)
                    except Exception:
                        pass


# ── Context builders ──────────────────────────────────────────────────────────

def _build_rag_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "Không có tài liệu liên quan."
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}. {c['section_title']}]\n{c['content']}")
    return "\n\n".join(parts)


def _build_system_with_context(
    chunks:          List[Dict[str, Any]],
    disease_key:     Optional[str],
    yolo_confidence: Optional[float],
) -> str:
    vi_name  = DISEASE_VI_NAMES.get(disease_key, "chưa xác định") if disease_key else "chưa xác định"
    conf_str = f" (conf={yolo_confidence:.1%})" if yolo_confidence else ""
    detection = (f"YOLO phát hiện: {vi_name}{conf_str}"
                 if disease_key else "YOLO chưa xác định bệnh.")
    context = _build_rag_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== TÀI LIỆU THAM KHẢO (hệ thống cung cấp) ===\n"
        f"{detection}\n\n{context}\n"
        f"=== HẾT TÀI LIỆU ==="
    )


def _build_messages(history: List[Dict], user_query: str, max_turns: int = 4) -> List[Dict]:
    recent   = history[-(max_turns * 2):] if history else []
    messages = [{"role": m["role"], "content": m["content"]} for m in recent]
    messages.append({"role": "user", "content": user_query})
    return messages


# ── Backend: Groq với sentence streaming ─────────────────────────────────────

def _generate_groq(
    user_query:  str,
    chunks:      List[Dict[str, Any]],
    disease_key: Optional[str],
    yolo_conf:   Optional[float],
    history:     List[Dict],
    voice_mode:  bool = False,
) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    client     = Groq(api_key=GROQ_API_KEY)
    system_ctx = _build_system_with_context(chunks, disease_key, yolo_conf)
    messages   = _build_messages(history, user_query)

    stream = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = [{"role": "system", "content": system_ctx}] + messages,
        max_tokens  = 400,
        temperature = 0.3,
        stream      = True,
        #reasoning_effort = "low",
    )

    print("\n🤖  ", end="", flush=True)

    tts_q      = _TTSQueue(voice_mode)
    all_tokens : List[str] = []
    sentence_buf = ""

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue

        print(token, end="", flush=True)
        all_tokens.append(token)
        sentence_buf += token

        # Phát hiện ranh giới câu → đẩy vào TTS queue ngay
        # Điều kiện: cuối câu (. ! ?) + đủ dài
        if (re.search(r'[.!?]\s*$', sentence_buf) and
                len(sentence_buf.strip()) >= _MIN_SENTENCE):
            tts_q.push(sentence_buf.strip())
            sentence_buf = ""

    # Phần còn lại chưa gửi
    if sentence_buf.strip():
        tts_q.push(sentence_buf.strip())

    # Chờ TTS đọc hết
    tts_q.finish()

    print()  # newline
    return "".join(all_tokens)


# ── Backend: none ─────────────────────────────────────────────────────────────

def _generate_no_llm(
    user_query:  str,
    chunks:      List[Dict[str, Any]],
    disease_key: Optional[str],
    yolo_conf:   Optional[float],
    history:     List[Dict],
    voice_mode:  bool = False,
) -> str:
    vi_name = DISEASE_VI_NAMES.get(disease_key, "Chưa xác định") if disease_key else "Chưa xác định"
    lines   = [f"[{vi_name}]\n", f"Câu hỏi: {user_query}\n"]
    seen    = set()
    for c in chunks:
        sec = c["section_title"]
        if sec in seen:
            continue
        seen.add(sec)
        lines.append(f"\n▶ {sec}\n" + "-" * 40)
        raw   = c["content"].split("\n")
        start = next((j + 1 for j, l in enumerate(raw)
                      if l.startswith("Bệnh:") or l.startswith("Chủ đề:")), 0)
        lines.append("\n".join(raw[start:]).strip())
    lines.append("\n⚕️  Vui lòng gặp bác sĩ da liễu.")
    result = "\n".join(lines)
    print(f"\n🤖  {result}")
    if voice_mode:
        from audio.tts import speak
        speak(result)
    return result


# ── Backend: Ollama subprocess ────────────────────────────────────────────────

def _generate_ollama(
    user_query:  str,
    chunks:      List[Dict[str, Any]],
    disease_key: Optional[str],
    yolo_conf:   Optional[float],
    history:     List[Dict],
    voice_mode:  bool = False,
) -> str:
    system_ctx   = _build_system_with_context(chunks, disease_key, yolo_conf)
    history_text = ""
    if history:
        parts = []
        for m in history[-(4 * 2):]:
            role    = "Trợ lý" if m["role"] == "assistant" else "Bệnh nhân"
            content = m["content"][:300] + "..." if len(m["content"]) > 300 else m["content"]
            parts.append(f"{role}: {content}")
        history_text = "\n\n--- LỊCH SỬ ---\n" + "\n\n".join(parts) + "\n--- HẾT ---\n\n"

    full_prompt = f"{system_ctx}\n\n{history_text}Bệnh nhân: {user_query}"

    print("\n🤖  ", end="", flush=True)
    tts_q        = _TTSQueue(voice_mode)
    tokens       : List[str] = []
    sentence_buf = ""

    proc = subprocess.Popen(
        ["ollama", "run", "--nowordwrap", OLLAMA_MODEL, full_prompt],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        tokens.append(line)
        sentence_buf += line
        if (re.search(r'[.!?]\s*$', sentence_buf) and
                len(sentence_buf.strip()) >= _MIN_SENTENCE):
            tts_q.push(sentence_buf.strip())
            sentence_buf = ""

    if sentence_buf.strip():
        tts_q.push(sentence_buf.strip())
    tts_q.finish()

    proc.wait()
    print()
    return "".join(tokens).strip()


# ── Backend: Anthropic ────────────────────────────────────────────────────────

def _generate_anthropic(
    user_query:  str,
    chunks:      List[Dict[str, Any]],
    disease_key: Optional[str],
    yolo_conf:   Optional[float],
    history:     List[Dict],
    voice_mode:  bool = False,
) -> str:
    import anthropic as _ant
    client     = _ant.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_ctx = _build_system_with_context(chunks, disease_key, yolo_conf)
    messages   = _build_messages(history, user_query)
    resp       = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=1024,
        system=system_ctx, messages=messages,
    )
    result = resp.content[0].text
    print(f"\n🤖  {result}")
    if voice_mode:
        tts_q = _TTSQueue(True)
        for s in _split_sentences(result):
            tts_q.push(s)
        tts_q.finish()
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def generate_response(
    user_query:       str,
    retrieved_chunks: List[Dict[str, Any]],
    disease_key:      Optional[str],
    yolo_confidence:  Optional[float] = None,
    history:          Optional[List[Dict]] = None,
    voice_mode:       bool = False,        # ← tham số mới
) -> str:
    history = history or []

    if disease_key in DISEASES_WITHOUT_RAG:
        vi_name = DISEASE_VI_NAMES.get(disease_key, disease_key)
        msg = (f"Phát hiện: {vi_name}\n\n"
               f"Chưa có dữ liệu y tế về bệnh này.\n"
               f"Vui lòng tham khảo bác sĩ da liễu.")
        print(f"\n🤖  {msg}")
        if voice_mode:
            from audio.tts import speak
            speak(msg)
        return msg

    backend = LLM_BACKEND.lower()
    kwargs  = dict(
        user_query  = user_query,
        chunks      = retrieved_chunks,
        disease_key = disease_key,
        yolo_conf   = yolo_confidence,
        history     = history,
        voice_mode  = voice_mode,
    )

    if backend == "groq":
        return _generate_groq(**kwargs)
    elif backend == "ollama":
        return _generate_ollama(**kwargs)
    elif backend == "anthropic":
        return _generate_anthropic(**kwargs)
    else:
        return _generate_no_llm(**kwargs)
