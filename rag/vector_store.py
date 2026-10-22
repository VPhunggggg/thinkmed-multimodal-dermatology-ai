"""
ChromaDB vector store: embed và lưu các DiseaseChunk.

Embedding model: intfloat/multilingual-e5-small
  - Hỗ trợ tiếng Việt tốt
  - ~470MB, chạy CPU (~1-2s/batch) hoặc GPU (nhanh hơn ~5x)
  - Format: "passage: <text>" khi index, "query: <text>" khi search

ChromaDB: persistent storage trên disk → không cần rebuild mỗi lần khởi động
"""

from typing import List
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from rag.document_processor import DiseaseChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_embedding_function():
    """Tạo embedding function (tự chọn CPU/GPU)."""
    device = "cpu"  # ÉP CPU: nhường VRAM cho YOLO. Embedder e5-small trên CPU chỉ ~200ms/query, thừa nhanh cho kiosk.
    print(f"[VectorStore] Embedding device: {device}")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=device,
    )


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection(client, ef):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── Main ingest function ──────────────────────────────────────────────────────

def ingest_chunks(chunks: List[DiseaseChunk], reset: bool = False) -> None:
    """
    Embed và lưu chunks vào ChromaDB.

    Args:
        chunks: Output của process_document()
        reset:  True → xóa collection cũ và build lại từ đầu
    """
    ef     = get_embedding_function()
    client = get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[VectorStore] Đã xóa collection cũ '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = get_collection(client, ef)

    # Dùng prefix "passage:" theo đặc tả của multilingual-e5
    documents = [f"passage: {c.content}" for c in chunks]
    ids       = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "disease_key":   c.disease_key,
            "disease_vi":    c.disease_vi,
            "section_title": c.section_title,
            "section_index": c.section_index,
        }
        for c in chunks
    ]

    # Upsert theo batch (tránh OOM trên Jetson)
    BATCH = 32
    for i in range(0, len(chunks), BATCH):
        collection.upsert(
            documents=documents[i:i+BATCH],
            ids=ids[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )
        print(f"  Indexed {min(i+BATCH, len(chunks))}/{len(chunks)} chunks...")

    total = collection.count()
    print(f"[VectorStore] ✅ Collection '{COLLECTION_NAME}': {total} chunks trong DB")


def get_collection_info() -> dict:
    """Trả về thông tin collection (dùng để kiểm tra trước khi query)."""
    try:
        client = get_client()
        ef     = get_embedding_function()
        col    = client.get_collection(COLLECTION_NAME, embedding_function=ef)
        return {"status": "ok", "count": col.count()}
    except Exception as e:
        return {"status": "error", "message": str(e)}
