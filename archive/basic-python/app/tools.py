import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from app.vectorstore import get_chunks as _get_chunks
from app.vectorstore import list_documents as _list_documents
from app.vectorstore import search


PAGE_CHARS = 1500  # per page, keeps the model's context small


def web_search(query: str, max_results: int = 3, read_top: int = 2) -> list[dict]:
    """Search the web (no API key). Returns title, url, snippet, plus page text for the top results."""
    last_error = None
    for backend in ("duckduckgo", "bing", "auto"):
        try:
            hits = DDGS(timeout=15).text(query, max_results=max_results, backend=backend)
            if hits:
                results = [
                    {"title": h["title"], "url": h["href"], "snippet": h["body"]}
                    for h in hits
                ]
                # Read the top pages ourselves: snippets alone rarely hold the answer.
                for r in results[:read_top]:
                    text = fetch_page(r["url"], max_chars=PAGE_CHARS)
                    if len(text) > 200 and not text.startswith("Could not fetch"):
                        r["content"] = text
                return results
        except Exception as e:  # engine blocked or timed out: try the next one
            last_error = e
    return [{"title": "Search failed", "url": "", "snippet": f"All engines failed: {last_error}"}]


def fetch_page(url: str, max_chars: int = 3000) -> str:
    """Download a web page and return its readable text (cut to max_chars)."""
    try:
        resp = httpx.get(
            url,
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"Could not fetch page: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:max_chars]


def list_documents() -> list[dict]:
    """List every stored document with its chunk count."""
    return _list_documents() or [{"info": "No documents are stored yet."}]


def get_chunks(source: str, limit: int = 3) -> list[dict]:
    """Show stored chunks of one document."""
    return _get_chunks(source, int(limit)) or [{"info": f"No chunks found for {source}."}]


REPORT_DIR = Path("data/reports")


def save_report(title: str, content: str) -> dict:
    """Save a finished report as a Markdown file. GATED: only runs after the user approves."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "report"
    name = f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.md"
    body = content.strip()
    if not body.startswith("# "):
        body = f"# {title}\n\n{body}"
    (REPORT_DIR / name).write_text(body + "\n", encoding="utf-8")
    return {"saved": name, "chars": len(body)}


TOP_K = 2


def search_documents(query: str) -> list[dict]:
    """Return the top-2 closest chunks, always. The model decides what is relevant."""
    if not _list_documents():
        return [{"info": "The user has not uploaded any documents. Try web_search instead."}]
    return [
        {"source": h["source"], "page": h["page"], "text": h["text"]}
        for h in search(query, TOP_K)
    ]
