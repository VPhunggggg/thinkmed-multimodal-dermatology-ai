"""
setup_rag.py — Chạy 1 lần để build ChromaDB từ file .docx

Usage:
    python setup_rag.py           # Build lần đầu
    python setup_rag.py --reset   # Xóa DB cũ và build lại
"""

import argparse
import sys
from pathlib import Path

from config import DOCX_PATH, CHROMA_PERSIST_DIR
from rag.document_processor import process_document
from rag.vector_store import ingest_chunks, get_collection_info


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB cho RAG skin disease")
    parser.add_argument("--reset", action="store_true", help="Xóa DB cũ và build lại")
    args = parser.parse_args()

    # Kiểm tra file docx
    if not Path(DOCX_PATH).exists():
        print(f"❌ Không tìm thấy: {DOCX_PATH}")
        print(f"   Copy file .docx vào: {DOCX_PATH}")
        sys.exit(1)

    print(f"📄 File: {DOCX_PATH}")
    print(f"💾 ChromaDB: {CHROMA_PERSIST_DIR}")
    print()

    # Kiểm tra DB hiện tại
    if not args.reset:
        info = get_collection_info()
        if info["status"] == "ok" and info["count"] > 0:
            print(f"✅ DB đã tồn tại với {info['count']} chunks.")
            print("   Dùng --reset để build lại từ đầu.")
            return

    # Process document
    chunks = process_document(DOCX_PATH)
    if not chunks:
        print("❌ Không tạo được chunk nào từ document!")
        sys.exit(1)

    print(f"\n📦 Ingest {len(chunks)} chunks vào ChromaDB...")
    ingest_chunks(chunks, reset=args.reset)

    # Verify
    info = get_collection_info()
    if info["status"] == "ok":
        print(f"\n✅ Hoàn tất! DB có {info['count']} chunks.")
    else:
        print(f"\n❌ Lỗi verify DB: {info['message']}")


if __name__ == "__main__":
    main()
