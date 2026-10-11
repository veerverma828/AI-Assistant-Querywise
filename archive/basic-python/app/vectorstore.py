from pathlib import Path

import chromadb
import ollama

UPLOAD_DIR = Path("data/uploads")

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 32

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into vectors with the local Ollama embedding model."""
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings


def add_chunks(chunks: list[dict]) -> int:
    """Embed chunks and store them. Re-uploading a file replaces all its old chunks."""
    for source in {c["source"] for c in chunks}:
        collection.delete(where={"source": source})
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = embed([f"search_document: {c['text']}" for c in batch])
        collection.upsert(
            ids=[f"{c['source']}-p{c['page']}-c{c['chunk_index']}" for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[
                {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
    return len(chunks)


def list_documents() -> list[dict]:
    """One row per stored file: name, number of chunks, pages covered."""
    metas = collection.get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for m in metas:
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


def delete_document(source: str) -> int:
    """Remove every stored chunk of one file. Returns how many chunks were removed."""
    ids = collection.get(where={"source": source}, include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_chunks(source: str, limit: int = 3) -> list[dict]:
    """First `limit` stored chunks of one file, in reading order."""
    result = collection.get(where={"source": source}, include=["documents", "metadatas"])
    rows = sorted(
        zip(result["documents"], result["metadatas"]),
        key=lambda r: (r[1]["page"], r[1]["chunk_index"]),
    )
    return [
        {"source": m["source"], "page": m["page"], "chunk_index": m["chunk_index"], "text": t}
        for t, m in rows[:limit]
    ]


def search(query: str, k: int = 2) -> list[dict]:
    """Return the k chunks closest in meaning to the query."""
    vector = embed([f"search_query: {query}"])
    result = collection.query(query_embeddings=vector, n_results=k)
    return [
        {
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "distance": round(dist, 4),
        }
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
