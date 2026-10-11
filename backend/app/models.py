"""The two models the app uses. Both are served by a local Ollama."""
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text was trained with a task prefix: one for stored text, one for questions."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        # The base embed_query() calls self.embed_documents(), which would add the document prefix
        # on top of this one. Call the base embed_documents() directly instead.
        return super().embed_documents([f"search_query: {text}"])[0]


def chat_model() -> ChatOllama:
    # Low temperature: the agent should follow the retrieved text and the tool rules, not improvise.
    return ChatOllama(model=CHAT_MODEL, temperature=0.1)


def embeddings() -> NomicEmbeddings:
    return NomicEmbeddings(model=EMBED_MODEL)
