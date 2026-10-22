"""
Patch này chỉ thay class _TTSQueue trong response_generator.py.
Chạy: python3 tts_queue_patch.py
"""

PATH = "/home/fptuadmin/Desktop/skin_disease_rag/rag/response_generator.py"

OLD = '''class _TTSQueue:
    """
    Background worker đọc câu từ queue và phát qua loa tuần tự.
    Dùng threading để LLM stream và TTS phát song song.
    """
    def __init__(self, voice_mode: bool):
        self.voice_mode = voice_mode
        self._q      = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def push(self, sentence: str):
        """Thêm câu vào queue để đọc."""
        if sentence.strip():
            self._q.put(sentence)

    def finish(self):
        """Đợi tất cả câu trong queue được đọc xong."""
        self._q.put(None)   # sentinel
        self._worker.join()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            if self.voice_mode:
                try:
                    from audio.tts import speak
                    speak(item)
                except Exception as e:
                    print(f"\\n[TTS] Lỗi: {e}", flush=True)
            self._q.task_done()'''

NEW = '''class _TTSQueue:
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
        VOICE = "vi-VN-HoaiMyNeural"

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
                print(f"\\n[TTS gen] Lỗi: {e}", flush=True)

    # ── Stage 2: MP3 → loa ───────────────────────────────────────────────────

    def _play_worker(self):
        import os, subprocess
        PLAYERS = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
            ["mpg123", "-q"],
        ]
        while True:
            item = self._audio_q.get()
            if item is self._END:
                break
            try:
                for cmd in PLAYERS:
                    try:
                        subprocess.run(cmd + [item], check=True, capture_output=True)
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
            except Exception as e:
                print(f"\\n[TTS play] Lỗi: {e}", flush=True)
            finally:
                try:
                    os.unlink(item)
                except Exception:
                    pass'''

content = open(PATH).read()

if OLD in content:
    open(PATH, "w").write(content.replace(OLD, NEW))
    print("✅ Patch _TTSQueue thành công")
elif "2-stage pipeline" in content:
    print("✅ Đã patch rồi — không cần làm gì")
else:
    print("❌ Không match — kiểm tra:")
    for i, line in enumerate(content.split('\n'), 1):
        if '_TTSQueue' in line or 'class _TTS' in line:
            print(f"  [{i}] {line}")
