"""
Retriever: query ChromaDB → trả về top-k chunks phù hợp.

Khi YOLO đã detect được bệnh, dùng metadata filter để chỉ search
trong bệnh đó (chính xác hơn, nhanh hơn).
Khi không có bệnh, search toàn bộ collection.
"""

from typing import Any, Dict, List, Optional
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb

from config import COLLECTION_NAME, TOP_K_CHUNKS
from rag.vector_store import get_client, get_embedding_function


class Retriever:
    def __init__(self):
        print("[Retriever] Kết nối ChromaDB...")
        self._ef         = get_embedding_function()
        self._client     = get_client()
        self._collection = self._client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
        )
        print(f"[Retriever] ✅ Collection '{COLLECTION_NAME}': {self._collection.count()} chunks")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        disease_key: Optional[str] = None,
        top_k: int = TOP_K_CHUNKS,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm chunks liên quan đến query_text.

        Args:
            query_text:  Câu hỏi đã được rewrite (tiếng Việt)
            disease_key: Lọc theo bệnh cụ thể nếu YOLO đã detect
            top_k:       Số chunks trả về

        Returns:
            List of {content, disease_key, disease_vi, section_title, score}
        """
        # Prefix "query:" theo đặc tả multilingual-e5
        prefixed = f"query: {query_text}"

        where = {"disease_key": {"$eq": disease_key}} if disease_key else None

        n = min(top_k, self._collection.count())
        results = self._collection.query(
            query_texts=[prefixed],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Bỏ prefix "passage: " đã thêm khi index
            clean = doc[len("passage: "):] if doc.startswith("passage: ") else doc
            chunks.append({
                "content":       clean,
                "disease_key":   meta["disease_key"],
                "disease_vi":    meta["disease_vi"],
                "section_title": meta["section_title"],
                "score":         round(1 - dist, 4),   # cosine distance → similarity
            })

        return chunks

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Chuyển list chunks thành chuỗi context cho LLM."""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Tài liệu {i} | {c['disease_vi']} — {c['section_title']}]\n"
                f"{c['content']}"
            )
        return "\n\n".join(parts)