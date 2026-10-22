"""
Parse tài liệu bệnh da (.docx) → DiseaseChunk list cho RAG.

Đọc TRỰC TIẾP docx bằng python-docx (không cần lệnh 'extract-text').

Cấu trúc tài liệu:
  - 6 bệnh, mỗi bệnh bắt đầu tại một Heading 1 CỤ THỂ (xem DISEASE_START_MARKERS).
  - Các Heading 1 KHÁC (vd "Điều trị vẩy nến") là MỤC CON của bệnh đang mở.
  - Heading 2/3 mở section trong bệnh.
  - Section ngắn hơn MIN_CHUNK_LEN được gộp vào section trước.

Cài: pip install python-docx
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import DOCX_PATH, DISEASE_VI_NAMES, MIN_CHUNK_LEN

# ── Mốc BẮT ĐẦU mỗi bệnh: (chuỗi nhận dạng trong Heading 1, canonical key) ─────
# So khớp sau khi bỏ dấu + lowercase → bền với khác biệt dấu/hoa thường.
# Thứ tự QUAN TRỌNG: đặc trưng hơn trước.
DISEASE_START_MARKERS = [
    ("viem da co dia",       "atopic dermatitis"),
    ("vay nen (psoriasis)",  "psoriasis"),
    ("benh phong (leprosy)", "leprosy"),
    ("mun coc do virus",     "wart"),
    ("mun trung ca (acne)",  "acne"),
    ("day sung nang long",   "keratosis"),
]


@dataclass
class DiseaseChunk:
    chunk_id:      str
    disease_key:   str
    disease_vi:    str
    section_title: str
    section_index: int
    content:       str
    raw_content:   str


def _strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase → so khớp bền."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower().strip()


def _match_disease_start(heading_text: str) -> Optional[str]:
    """Heading 1 này có phải MỐC BẮT ĐẦU của một bệnh không?"""
    norm = _strip_accents(heading_text)
    for marker, key in DISEASE_START_MARKERS:
        if norm.startswith(marker):
            return key
    return None


def _clean(text: str) -> str:
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)   # [text](url) → text
    text = re.sub(r'https?://\S+', '', text)                # xóa URL rời
    return text.strip()


def _merge_short_sections(sections, min_len=MIN_CHUNK_LEN):
    if not sections:
        return sections
    merged = [list(sections[0])]
    for title, content in sections[1:]:
        prev_title, prev_content = merged[-1]
        if len(prev_content.strip()) < min_len:
            sep = f"\n\n▶ {title}\n" if title else "\n\n"
            merged[-1][1] = prev_content + sep + content
        else:
            merged.append([title, content])
    return [tuple(m) for m in merged]


def process_document(docx_path: Path = DOCX_PATH) -> List[DiseaseChunk]:
    import docx
    print(f"[DocProcessor] Đọc file: {docx_path}")
    d = docx.Document(str(docx_path))

    # ── Pass 1: gom theo bệnh → theo section ──────────────────────────────────
    disease_blocks: "dict[str, list]" = {}   # key → list[[section_title, [lines]]]
    cur_key: Optional[str] = None
    cur_sections: Optional[list] = None

    for p in d.paragraphs:
        style = p.style.name
        text  = _clean(p.text)
        if not text and not style.startswith("Heading"):
            continue

        if style == "Heading 1":
            key = _match_disease_start(text)
            if key:
                # Mốc bắt đầu BỆNH MỚI
                cur_key = key
                cur_sections = disease_blocks.setdefault(key, [])
                cur_sections.append([text, []])
                continue
            # H1 khác → mục con của bệnh đang mở
            if cur_sections is not None:
                cur_sections.append([text, []])
            continue

        if cur_key is None:
            continue   # nội dung trước bệnh đầu tiên

        if style.startswith("Heading"):        # Heading 2/3 → section mới
            cur_sections.append([text, []])
        else:
            if not text:
                continue
            if not cur_sections:
                cur_sections.append(["", []])
            cur_sections[-1][1].append(text)

    # ── Pass 2: dựng DiseaseChunk ─────────────────────────────────────────────
    all_chunks: List[DiseaseChunk] = []
    for key, sections in disease_blocks.items():
        vi_name = DISEASE_VI_NAMES.get(key, key)
        secs = [(t, "\n".join(lines).strip()) for t, lines in sections]
        secs = [(t, c) for t, c in secs if c or t]
        secs = _merge_short_sections(secs)

        idx = 0
        for sec_title, sec_content in secs:
            if not sec_content.strip():
                continue
            content = (
                f"Bệnh: {vi_name} ({key})\n"
                f"Chủ đề: {sec_title}\n\n"
                f"{sec_content}"
            )
            all_chunks.append(DiseaseChunk(
                chunk_id      = f"{key.replace(' ', '_')}__{idx:03d}",
                disease_key   = key,
                disease_vi    = vi_name,
                section_title = sec_title or "Tổng quan",
                section_index = idx,
                content       = content,
                raw_content   = sec_content,
            ))
            idx += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"[DocProcessor] Tổng: {len(all_chunks)} chunks từ {len(disease_blocks)} bệnh")
    for key in disease_blocks:
        n = sum(1 for c in all_chunks if c.disease_key == key)
        print(f"  • {DISEASE_VI_NAMES.get(key, key):<40} {n} chunks")
    return all_chunks


if __name__ == "__main__":
    chunks = process_document()
    print("\n=== Mẫu 2 chunk đầu ===")
    for c in chunks[:2]:
        print(f"\n[{c.chunk_id}] {c.disease_vi} — {c.section_title}")
        print(c.raw_content[:200])
