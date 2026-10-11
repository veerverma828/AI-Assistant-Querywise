"""The vector store (Chroma, through LangChain) and the document bookkeeping the UI needs."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.models import embeddings

UPLOAD_DIR = Path("data/uploads")
BATCH_SIZE = 32  # chunks embedded per Ollama call

store = Chroma(
    collection_name="documents",
    embedding_function=embeddings(),
    persist_directory="data/chroma",
    collection_metadata={"hnsw:space": "cosine"},
)


def _chunk_id(doc: Document) -> str:
    m = doc.metadata
    return f"{m['source']}-p{m['page']}-c{m['chunk_index']}"


def add_documents(docs: list[Document]) -> int:
    """Embed and store chunks. Re-uploading a file replaces all of its old chunks."""
    for source in {d.metadata["source"] for d in docs}:
        delete_document(source)
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        store.add_documents(batch, ids=[_chunk_id(d) for d in batch])
    return len(docs)


def search(query: str, k: int) -> list[Document]:
    """The k chunks closest in meaning to the query."""
    return store.similarity_search(query, k=k)


def list_documents() -> list[dict]:
    """One row per stored file: name, number of chunks, pages covered, and whether the file is still on disk."""
    docs: dict[str, dict] = {}
    for m in store.get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["source"], {"source": m["source"], "chunks": 0, "pages": set()})
        d["chunks"] += 1
        d["pages"].add(m["page"])
    return [
        {
            "source": d["source"],
            "chunks": d["chunks"],
            "pages": sorted(d["pages"]),
            "file_on_disk": (UPLOAD_DIR / d["source"]).is_file(),
        }
        for d in sorted(docs.values(), key=lambda d: d["source"])
    ]


def get_chunks(source: str, limit: int = 3) -> list[dict]:
    """First `limit` stored chunks of one file, in reading order."""
    result = store.get(where={"source": source}, include=["documents", "metadatas"])
    rows = sorted(
        zip(result["documents"], result["metadatas"]),
        key=lambda r: (r[1]["page"], r[1]["chunk_index"]),
    )
    return [
        {"source": m["source"], "page": m["page"], "chunk_index": m["chunk_index"], "text": t}
        for t, m in rows[:limit]
    ]


def delete_document(source: str) -> int:
    """Remove every stored chunk of one file. Returns how many chunks were removed."""
    ids = store.get(where={"source": source}, include=[])["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)
