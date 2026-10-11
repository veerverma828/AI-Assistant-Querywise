import ollama

from app.vectorstore import search

LLM_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context given by the user. "
    "Write a complete answer in 1 to 3 full sentences. "
    "After each fact, add the source number in brackets, like [1] or [2]. "
    "Never reply with only a citation. "
    "If the context does not contain the answer, reply exactly: I don't know based on the documents."
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        label = f"[{number}] ({chunk['source']}, p.{chunk['page']})"
        blocks.append(label + "\n" + chunk["text"])

    context = "\n\n".join(blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in full sentences using only the context, with [number] citations:"
    )


def answer_question(question: str, k: int = 2) -> dict:
    chunks = search(question, k)

    if not chunks:  # empty vector store: nothing to send
        return {
            "answer": "No documents are stored yet. Upload a document first.",
            "sources": [],
        }

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        options={"temperature": 0.1},
    )

    sources = [{"id": i, **chunk} for i, chunk in enumerate(chunks, start=1)]
    return {"answer": response.message.content, "sources": sources}
