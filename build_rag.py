#!/usr/bin/env python3
"""
build_rag.py — Build lại ChromaDB từ tài liệu mới (skin_old_new_docu.docx).

VẤN ĐỀ CẦN XỬ LÝ (không phải chỉ nạp thẳng):

1. FOLDER-STRUCTURE ≠ DISEASE. Tài liệu có 27 khối Heading 1, nhưng model chỉ
   phân loại 5 bệnh (acne/atopic_dermatitis/psoriasis/tinea/wart). Phải GOM
   nhiều khối H1 về đúng một disease_key:
     - "Viêm da cơ địa" + "Nguyên nhân..." + "Điều trị..." -> atopic_dermatitis
     - 10 khối "Nấm ..." (đầu/thân/bẹn/chân/móng...) -> tinea
   disease_key PHẢI khớp CLASS_TO_DISEASE của classifier để retriever lọc đúng.

2. BỆNH NGOÀI PHẠM VI CHẨN ĐOÁN ẢNH. Một số khối được giữ cho TẦNG HỘI THOẠI
   (bệnh nhân mô tả triệu chứng bằng lời), KHÔNG phải chẩn đoán từ ảnh:
     - leprosy (phong), keratosis_pilaris (dày sừng nang lông), impetigo (chốc)
   Chúng vào ChromaDB với disease_key riêng, nhưng model KHÔNG bao giờ trỏ tới
   -> chỉ truy cập được khi user tự nhắc tên/triệu chứng (qua DISEASE_ALIASES).

3. KHỐI BỊ LOẠI HOÀN TOÀN. HPV sinh dục / ung thư cổ tử cung / vắc-xin HPV:
   nhạy cảm, không hợp kiosk công cộng, model không phân loại. BỎ.

Chạy:
    pip install python-docx chromadb sentence-transformers
    python build_rag.py
"""

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import docx
except ImportError:
    sys.exit("pip install python-docx")

# ══════════════════════════════════════════════════════════════════════
#  ÁNH XẠ KHỐI HEADING 1 -> disease_key
# ══════════════════════════════════════════════════════════════════════
# Khớp theo tiền tố text của Heading 1 (đã .strip().lower()).
# disease_key khớp CLASS_TO_DISEASE của classifier + vài key "hội thoại".

H1_TO_KEY = [
    # (chuỗi bắt đầu Heading 1 — lowercase,  disease_key)

    # ── 5 bệnh model CHẨN ĐOÁN được (khớp classifier) ──
    ("viêm da cơ địa",          "atopic_dermatitis"),
    ("nguyên nhân của viêm da",  "atopic_dermatitis"),
    ("ảnh hưởng tâm lý của viêm da", "atopic_dermatitis"),
    ("điều trị viêm da cơ địa",  "atopic_dermatitis"),

    ("vẩy nến",                  "psoriasis"),
    ("điều trị vẩy nến",         "psoriasis"),
    ("ảnh hưởng tâm lý của vẩy nến", "psoriasis"),
    ("điều trị vẩy nến trong thai kỳ", "psoriasis"),

    ("mụn cóc do virus",         "wart"),

    ("mụn trứng cá",             "acne"),
    ("10 thói quen",             "acne"),
    ("kem dưỡng ẩm",             "acne"),
    ("mụn trứng cá có thể ảnh hưởng", "acne"),

    ("bệnh nấm da",              "tinea"),
    ("nấm râu",                  "tinea"),
    ("nấm da đầu",               "tinea"),
    ("nấm da thân",              "tinea"),
    ("nấm bẹn",                  "tinea"),
    ("nấm da mặt",               "tinea"),
    ("nấm bàn tay",              "tinea"),
    ("nấm bàn chân",             "tinea"),
    ("nhiễm nấm móng",           "tinea"),

    # ── Bệnh chỉ dùng ở TẦNG HỘI THOẠI (model không trỏ tới) ──
    ("bệnh phong",               "leprosy"),
    ("dày sừng nang lông",       "keratosis_pilaris"),
    ("chốc lở",                  "impetigo"),
]

# Khối BỊ LOẠI HOÀN TOÀN (khớp tiền tố -> bỏ)
DROP_PREFIXES = [
    "hpv lây qua đường tình dục",
    "nhiễm virus u nhú ở người không lây",
    "vắc-xin phòng virus u nhú",
]

# Tên tiếng Việt hiển thị cho mỗi disease_key
KEY_VI = {
    "atopic_dermatitis": "Viêm da cơ địa",
    "psoriasis":         "Vẩy nến",
    "wart":              "Mụn cóc do virus",
    "acne":              "Mụn trứng cá",
    "tinea":             "Nấm da",
    "leprosy":           "Bệnh phong",
    "keratosis_pilaris": "Dày sừng nang lông",
    "impetigo":          "Chốc lở",
}

# disease_key nào model CÓ thể chẩn đoán từ ảnh (khớp classifier)
IMAGE_CLASSIFIABLE = {"atopic_dermatitis", "psoriasis", "wart", "acne", "tinea"}

MIN_CHUNK_LEN = 150   # ký tự — bỏ chunk quá ngắn


def key_for_heading(text: str):
    """Trả về (disease_key | None, is_dropped)."""
    t = text.strip().lower()
    for pref in DROP_PREFIXES:
        if t.startswith(pref):
            return None, True
    for pref, key in H1_TO_KEY:
        if t.startswith(pref):
            return key, False
    return None, False


def extract_chunks(docx_path: Path):
    """
    Cắt tài liệu thành chunk theo Heading 2/3 (mỗi mục = 1 chunk),
    gắn disease_key theo Heading 1 gần nhất phía trên.
    """
    doc = docx.Document(docx_path)
    paras = doc.paragraphs

    chunks = []
    cur_key = None
    cur_dropped = False
    cur_section = ""
    buf = []
    stats = {}
    dropped_headings = []

    def flush():
        nonlocal buf
        if cur_key and not cur_dropped and buf:
            text = "\n".join(buf).strip()
            if len(text) >= MIN_CHUNK_LEN:
                chunks.append({
                    "disease_key":   cur_key,
                    "disease_vi":    KEY_VI[cur_key],
                    "section_title": cur_section or KEY_VI[cur_key],
                    "content":       text,
                })
                stats[cur_key] = stats.get(cur_key, 0) + 1
        buf = []

    for p in paras:
        t = p.text.strip()
        st = p.style.name

        if st == "Heading 1":
            flush()
            k, dropped = key_for_heading(t)
            cur_key, cur_dropped = k, dropped
            cur_section = t
            if dropped:
                dropped_headings.append(t)
            continue

        if st in ("Heading 2", "Heading 3", "Heading 4"):
            flush()
            cur_section = t
            continue

        if t:
            buf.append(t)

    flush()
    return chunks, stats, dropped_headings


def main():
    src = Path("Desktop/skin_disease_rag/data/skin_old_new_docu.docx")
    if not src.exists():
        # đường dẫn khi chạy trên máy bạn
        src = Path(__file__).parent / "data" / "skin_old_new_docu.docx"
    if not src.exists():
        sys.exit(f"Không tìm thấy tài liệu: {src}")

    print("=" * 66)
    print("BUILD RAG — cắt chunk + gom theo disease_key")
    print("=" * 66)

    chunks, stats, dropped = extract_chunks(src)

    print(f"\nTổng chunk hợp lệ: {len(chunks)}\n")
    print(f"{'disease_key':22s} {'chunk':>6s}  {'model chẩn đoán?':>16s}")
    print("-" * 50)
    for k in sorted(stats):
        img = "✓ (từ ảnh)" if k in IMAGE_CLASSIFIABLE else "chỉ hội thoại"
        print(f"{k:22s} {stats[k]:6d}  {img:>16s}")

    print(f"\nĐã LOẠI {len(dropped)} khối (ngoài phạm vi / nhạy cảm):")
    for h in dropped:
        print(f"   ✗ {h[:60]}")

    # Kiểm tra: 5 lớp model có đủ chunk không
    print("\n" + "=" * 66)
    missing = IMAGE_CLASSIFIABLE - set(stats)
    if missing:
        print(f"⚠ THIẾU chunk cho lớp model: {missing}")
    else:
        print("✓ Cả 5 lớp model đều có tài liệu RAG.")

    # Ghi ra JSONL để kiểm tra trước khi nạp ChromaDB
    import json
    out = Path("rag_chunks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n-> {out} ({len(chunks)} chunk) — kiểm tra trước khi nạp ChromaDB")

    # ── Nạp ChromaDB (bỏ comment khi đã chạy được trên máy có chromadb) ──
    print("\n" + "=" * 66)
    print("BƯỚC NẠP CHROMADB")
    print("=" * 66)


    import chromadb
    from chromadb.utils import embedding_functions
    from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)   # xoá collection cũ (5 bệnh cũ)
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="cpu")
    col = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    import json
    rows = [json.loads(l) for l in open("rag_chunks.jsonl", encoding="utf-8")]
    col.add(
        ids       =[f"c{i}" for i in range(len(rows))],
        documents =[f"passage: {r['content']}" for r in rows],   # prefix e5
        metadatas =[{"disease_key": r["disease_key"],
                     "disease_vi":  r["disease_vi"],
                     "section_title": r["section_title"]} for r in rows],
    )
    print(f"ChromaDB: {col.count()} chunks")


if __name__ == "__main__":
    main()
