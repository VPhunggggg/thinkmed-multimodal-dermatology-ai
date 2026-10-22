"""
Response Generator — streaming TTS version.
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
Bạn là Bác sĩ Da liễu ThinkMed. Hãy tư vấn SIÊU NGẮN GỌN, BẮT ĐẦU NGAY BẰNG GẠCH ĐẦU DÒNG (- ).

Quy tắc BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG CHÀO DÀI DÒNG, KHÔNG LẶP LẠI CÂU HỎI: Không nói 'Chào bạn', 'Về câu hỏi của bạn...'. Đi thẳng ngay vào các ý chính bằng gạch đầu dòng '- '.
2. ĐỊNH DẠNG GẠCH ĐẦU DÒNG (- ): Trình bày tối đa 3-4 dòng gạch đầu dòng '- ' ngắn gọn. Mỗi dòng chỉ từ 1 đến 2 câu súc tích.
3. DÙNG TIẾNG VIỆT THUẦN TÚY: Luôn dùng tên tiếng Việt (Vẩy nến, Nấm da, Mụn trứng cá, Viêm da cơ địa, Kháng sinh). Hạn chế tối đa từ tiếng Anh chuyên ngành.
4. NỘI DUNG TRỌNG TÂM:
   - Ý 1: Trả lời trực diện & giải thích ngắn.
   - Ý 2: Hướng dẫn chăm sóc tại nhà.
   - Ý 3: Khi nào cần gặp bác sĩ khám trực tiếp.\
"""

_SENTENCE_END = re.compile(r'(?<=[.!?:])\s+|(?<=\n)\s*')
_MIN_SENTENCE  = 35


def _split_sentences(text: str) -> list[str]:
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
    _END = object()

    def __init__(self, voice_mode: bool, stop_event: Optional[threading.Event] = None):
        import time
        self.voice_mode = voice_mode
        self.stop_event = stop_event
        self.t_start    = time.time()
        self._text_q    = queue.Queue()
        self._audio_q   = queue.Queue()
        if voice_mode:
            self._gen_t  = threading.Thread(target=self._gen_worker,  daemon=True)
            self._play_t = threading.Thread(target=self._play_worker, daemon=True)
            self._gen_t.start()
            self._play_t.start()

    def push(self, sentence: str):
        if self.stop_event and self.stop_event.is_set():
            return
        if sentence.strip() and self.voice_mode:
            self._text_q.put(sentence.strip())

    def finish(self):
        if self.voice_mode:
            if self.stop_event and self.stop_event.is_set():
                while not self._text_q.empty():
                    try: self._text_q.get_nowait()
                    except Exception: break
                while not self._audio_q.empty():
                    try:
                        p = self._audio_q.get_nowait()
                        if isinstance(p, str) and os.path.exists(p):
                            try: os.unlink(p)
                            except Exception: pass
                    except Exception: break
                return
            self._text_q.put(self._END)
            self._gen_t.join()
            self._play_t.join()

    def _gen_worker(self):
        import asyncio, tempfile, os
        from audio.tts import synthesize_pcm, synthesize_elevenlabs, TTS_BACKEND
        VOICE = "vi-VN-HoaiMyNeural"
        RATE  = "+10%"

        async def _to_mp3(text: str, path: str):
            import edge_tts, re
            raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
            if not raw_sentences:
                raw_sentences = [text.strip()] if text.strip() else []
            if not raw_sentences:
                return

            from audio.tts import _clean_text_for_tts
            sentences = []
            for s in raw_sentences:
                s = _clean_text_for_tts(s)
                if not s.endswith(('.', '!', '?', ':')):
                    s = s + '.'
                s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
                sentences.append(s)

            combined_data = bytearray()
            voices_to_try = [VOICE, "vi-VN-NamMinhNeural"]

            for s in sentences:
                if self.stop_event and self.stop_event.is_set():
                    return
                sent_data = bytearray()
                for v in voices_to_try:
                    for attempt in range(2):
                        try:
                            communicate = edge_tts.Communicate(s, v, rate=RATE)
                            async for chunk in communicate.stream():
                                if self.stop_event and self.stop_event.is_set():
                                    return
                                if chunk["type"] == "audio":
                                    sent_data.extend(chunk["data"])
                            if sent_data:
                                break
                        except Exception:
                            await asyncio.sleep(0.15)
                    if sent_data:
                        break
                if sent_data:
                    combined_data.extend(sent_data)

            if not combined_data:
                raise RuntimeError("No audio data generated")

            with open(path, "wb") as f:
                f.write(combined_data)

        while True:
            if self.stop_event and self.stop_event.is_set():
                break
            try:
                item = self._text_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is self._END:
                self._audio_q.put(self._END)
                break
            if not item or not item.strip():
                continue

            # 1. Thử ElevenLabs nếu cấu hình là "elevenlabs"
            if TTS_BACKEND == "elevenlabs":
                try:
                    tmp_el = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    el_path = tmp_el.name
                    tmp_el.close()
                    ok = synthesize_elevenlabs(item, el_path, stop_event=self.stop_event)
                    if ok and os.path.exists(el_path):
                        if self.stop_event and self.stop_event.is_set():
                            try: os.unlink(el_path)
                            except Exception: pass
                            break
                        self._audio_q.put(("elevenlabs", el_path))
                        continue
                    else:
                        if os.path.exists(el_path):
                            try: os.unlink(el_path)
                            except Exception: pass
                except Exception as e:
                    print(f"\n[TTS ElevenLabs worker] Lỗi: {e}", flush=True)

            # 2. Thử Piper TTS trực tiếp trên RAM nếu bật hoặc khi ElevenLabs fallback (Offline 100%)
            if TTS_BACKEND in ("elevenlabs", "piper"):
                try:
                    pcm_res = synthesize_pcm(item)
                    if pcm_res is not None:
                        if self.stop_event and self.stop_event.is_set():
                            break
                        self._audio_q.put(("piper", pcm_res))
                        continue
                except Exception as e:
                    print(f"\n[TTS Piper worker] Lỗi: {e}", flush=True)

            # 3. Fallback sang Edge-TTS qua MP3 tạm
            try:
                tmp  = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                path = tmp.name
                tmp.close()
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_to_mp3(item, path))
                finally:
                    try:
                        pending = asyncio.all_tasks(loop)
                        for t in pending:
                            t.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        pass
                    try:
                        loop.close()
                    except Exception:
                        pass
                if self.stop_event and self.stop_event.is_set():
                    if os.path.exists(path):
                        try: os.unlink(path)
                        except Exception: pass
                    break
                self._audio_q.put(("edge", path))
            except Exception as e:
                print(f"\n[TTS gen] Lỗi: {e}", flush=True)

    def _play_worker(self):
        import os, time
        from audio.tts import _pad_silence, play_audio, play_audio_array
        first = True
        while True:
            if self.stop_event and self.stop_event.is_set():
                break
            try:
                item = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is self._END:
                break

            engine, payload = item
            if first:
                dt = time.time() - self.t_start
                engine_name = "ElevenLabs API" if engine == "elevenlabs" else ("Piper TTS (Offline)" if engine == "piper" else "Microsoft Edge-TTS (Hoài My)")
                print(f"\n   [TTFA] Loa bắt đầu phát sau: {dt:.2f}s [{engine_name}]", flush=True)
                first = False

            if engine == "piper":
                # Piper TTS in-memory (audio_data, sample_rate)
                audio_data, sample_rate = payload
                try:
                    play_audio_array(audio_data, sample_rate, stop_event=self.stop_event)
                except Exception as e:
                    print(f"\n[TTS play array] Lỗi: {e}", flush=True)
            else:
                # File MP3 từ ElevenLabs hoặc Edge-TTS
                play = _pad_silence(payload) if first else payload
                try:
                    play_audio(play, stop_event=self.stop_event)
                except Exception as e:
                    print(f"\n[TTS play] Lỗi: {e}", flush=True)
                finally:
                    for _p in {payload, play}:
                        if os.path.exists(_p):
                            try:
                                os.unlink(_p)
                            except Exception:
                                pass


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
    voice_mode:      bool = False,
) -> str:
    vi_name  = DISEASE_VI_NAMES.get(disease_key, "chưa xác định") if disease_key else "chưa xác định"
    conf_str = f" (conf={yolo_confidence:.1%})" if yolo_confidence else ""
    detection = (f"Phát hiện: {vi_name}{conf_str}"
                 if disease_key else "Chưa xác định bệnh.")
    context = _build_rag_context(chunks)
    voice_rule = ""
    if voice_mode:
        voice_rule = (
            "\n\nLƯU Ý MÀN HÌNH & GIỌNG NÓI: Trả lời ngắn gọn, súc tích bằng 3-4 dòng gạch đầu dòng '- '. "
            "Các dòng gạch đầu dòng '- ' giúp người nhìn màn hình rất dễ đọc, giọng đọc phát loa sẽ tự động trôi chảy."
        )
    return (
        f"{SYSTEM_PROMPT}{voice_rule}\n\n"
        f"=== TÀI LIỆU THAM KHẢO (hệ thống cung cấp) ===\n"
        f"{detection}\n\n{context}\n"
        f"=== HẾT TÀI LIỆU ==="
    )


def _build_messages(history: List[Dict], user_query: str, max_turns: int = 4) -> List[Dict]:
    recent   = history[-(max_turns * 2):] if history else []
    messages = [{"role": m["role"], "content": m["content"]} for m in recent]
    messages.append({"role": "user", "content": user_query})
    return messages


def _clean_md(s: str) -> str:
    """Bỏ markdown để TTS không đọc 'hai sao'."""
    return s.replace("**", "").replace("##", "")


def _generate_groq(
    user_query:  str,
    chunks:      List[Dict[str, Any]],
    disease_key: Optional[str],
    yolo_conf:   Optional[float],
    history:     List[Dict],
    voice_mode:  bool = False,
    on_text_update: Optional[Any] = None,
    stop_event:  Optional[Any] = None,
) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    client     = Groq(api_key=GROQ_API_KEY)
    system_ctx = _build_system_with_context(chunks, disease_key, yolo_conf, voice_mode)
    messages   = _build_messages(history, user_query)

    max_tok = 220 if voice_mode else 450

    create_kwargs = dict(
        model       = GROQ_MODEL,
        messages    = [{"role": "system", "content": system_ctx}] + messages,
        max_tokens  = max_tok,
        temperature = 0.3,
        stream      = True,
    )
    # gpt-oss là reasoning model: giảm reasoning để dành token cho câu trả lời.
    # Nếu SDK Groq bản cũ không nhận, tự bỏ tham số này.
    try:
        if "gpt-oss" in GROQ_MODEL:
            stream = client.chat.completions.create(reasoning_effort="low", **create_kwargs)
        else:
            stream = client.chat.completions.create(**create_kwargs)
    except Exception:
        stream = client.chat.completions.create(**create_kwargs)

    print("\n🤖  ", end="", flush=True)

    tts_q        = _TTSQueue(voice_mode, stop_event=stop_event)
    all_tokens   : List[str] = []
    sentence_buf = ""
    is_first_chunk = True

    for chunk in stream:
        if stop_event and stop_event.is_set():
            break

        # Chunk có thể không có choices (metadata) hoặc content=None (reasoning).
        # Bỏ qua AN TOÀN, KHÔNG thoát vòng lặp.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", None) or ""
        if not token:
            continue

        print(token, end="", flush=True)
        all_tokens.append(token)
        sentence_buf += token

        # Cập nhật phụ đề tức thì lên màn hình Kiosk sau mỗi 2 từ
        if on_text_update and len(all_tokens) % 2 == 0:
            try:
                on_text_update(_clean_md("".join(all_tokens)).strip())
            except Exception:
                pass

        if stop_event and stop_event.is_set():
            break

        # Tối ưu TTFA: Câu đầu tiên ngắt khi gặp dấu ngắt câu (. ! ? : ; \n) hoặc dấu phẩy sau >= 25 ký tự
        if is_first_chunk:
            if (re.search(r'[.!?:;\n]\s*$', sentence_buf) and len(sentence_buf.strip()) >= 12) or \
               (re.search(r'[,]\s*$', sentence_buf) and len(sentence_buf.strip()) >= 25):
                tts_q.push(_clean_md(sentence_buf.strip()))
                sentence_buf = ""
                is_first_chunk = False
        else:
            # Các câu sau: ranh giới câu chuẩn (ngắt khi >= 20 ký tự)
            if ((re.search(r'[.!?:;\n]\s*$', sentence_buf))
                    and len(sentence_buf.strip()) >= 20):
                tts_q.push(_clean_md(sentence_buf.strip()))
                sentence_buf = ""

    if not (stop_event and stop_event.is_set()) and sentence_buf.strip():
        tts_q.push(_clean_md(sentence_buf.strip()))

    if on_text_update and not (stop_event and stop_event.is_set()):
        try:
            on_text_update(_clean_md("".join(all_tokens)).strip())
        except Exception:
            pass

    tts_q.finish()
    print()
    return _clean_md("".join(all_tokens)).strip()


def _generate_no_llm(
    user_query, chunks, disease_key, yolo_conf, history, voice_mode=False,
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


def _generate_ollama(
    user_query, chunks, disease_key, yolo_conf, history, voice_mode=False,
) -> str:
    system_ctx   = _build_system_with_context(chunks, disease_key, yolo_conf, voice_mode)
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
            tts_q.push(_clean_md(sentence_buf.strip()))
            sentence_buf = ""

    if sentence_buf.strip():
        tts_q.push(_clean_md(sentence_buf.strip()))
    tts_q.finish()

    proc.wait()
    print()
    return _clean_md("".join(tokens)).strip()


def _generate_anthropic(
    user_query, chunks, disease_key, yolo_conf, history, voice_mode=False,
) -> str:
    import anthropic as _ant
    client     = _ant.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_ctx = _build_system_with_context(chunks, disease_key, yolo_conf, voice_mode)
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
            tts_q.push(_clean_md(s))
        tts_q.finish()
    return result


def generate_response(
    user_query:       str,
    retrieved_chunks: List[Dict[str, Any]],
    disease_key:      Optional[str],
    yolo_confidence:  Optional[float] = None,
    history:          Optional[List[Dict]] = None,
    voice_mode:       bool = False,
    on_text_update:   Optional[Any] = None,
    stop_event:       Optional[Any] = None,
) -> str:
    history = history or []

    if stop_event and stop_event.is_set():
        return ""

    if disease_key in DISEASES_WITHOUT_RAG:
        vi_name = DISEASE_VI_NAMES.get(disease_key, disease_key)
        msg = (f"Phát hiện: {vi_name}\n\n"
               f"Chưa có dữ liệu y tế về bệnh này.\n"
               f"Vui lòng tham khảo bác sĩ da liễu.")
        print(f"\n🤖  {msg}")
        if on_text_update:
            on_text_update(msg)
        if voice_mode and not (stop_event and stop_event.is_set()):
            from audio.tts import speak
            speak(msg, stop_event=stop_event)
        return msg

    backend = LLM_BACKEND.lower()
    kwargs  = dict(
        user_query  = user_query,
        chunks      = retrieved_chunks,
        disease_key = disease_key,
        yolo_conf   = yolo_confidence,
        history     = history,
        voice_mode  = voice_mode,
        on_text_update = on_text_update,
        stop_event  = stop_event,
    )

    if backend == "groq":
        return _generate_groq(**kwargs)
    elif backend == "ollama":
        kwargs.pop("on_text_update", None)
        kwargs.pop("stop_event", None)
        return _generate_ollama(**kwargs)
    elif backend == "anthropic":
        kwargs.pop("on_text_update", None)
        kwargs.pop("stop_event", None)
        return _generate_anthropic(**kwargs)
    else:
        kwargs.pop("on_text_update", None)
        kwargs.pop("stop_event", None)
        return _generate_no_llm(**kwargs)
