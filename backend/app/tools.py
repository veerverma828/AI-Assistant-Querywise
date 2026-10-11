"""
The agent's tools. @tool turns each function and its docstring into the schema the model reads:
the summary is the tool description, and the "Args:" section describes the parameters.
"""
import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.types import interrupt

from app import store

TOP_K = 2  # passages returned by search_documents
PAGE_CHARS = 1500  # per web page, keeps the model's context small
REPORT_DIR = Path("data/reports")


def _page_text(url: str, max_chars: int) -> str:
    """Download a web page and return its readable text, cut to max_chars."""
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
    return " ".join(soup.get_text(" ").split())[:max_chars]


def write_report(title: str, content: str) -> dict:
    """Write a report to disk as Markdown. Used by the save_report tool and by POST /reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "report"
    name = f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.md"
    body = content.strip()
    if not body.startswith("# "):
        body = f"# {title}\n\n{body}"
    (REPORT_DIR / name).write_text(body + "\n", encoding="utf-8")
    return {"saved": name, "chars": len(body)}


@tool(parse_docstring=True)
def search_documents(query: str) -> list[dict]:
    """Search the user's own uploaded files (resume, PDFs, notes, reports).

    Use for any question about the user, their background, skills, projects, or content that could be
    in their files, even if the question does not mention documents. Not for news, weather, or public
    facts. Returns the 2 closest passages: read them, ignore any that do not help, and use web_search
    if none answer the question.

    Args:
        query: What to look for.
    """
    if not store.list_documents():
        return [{"info": "The user has not uploaded any documents. Try web_search instead."}]
    return [
        {"source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content}
        for d in store.search(query, TOP_K)
    ]


@tool
def list_documents() -> list[dict]:
    """List which documents are stored, with chunk count and pages for each.

    Use when asked what documents or files are stored or uploaded.
    """
    return store.list_documents() or [{"info": "No documents are stored yet."}]


@tool(parse_docstring=True)
def get_chunks(source: str, limit: int = 3) -> list[dict]:
    """Show the stored text chunks of one document, in order.

    Use when asked to show or inspect a document's chunks.

    Args:
        source: Exact file name, for example report.pdf.
        limit: How many chunks to show.
    """
    return store.get_chunks(source, limit) or [{"info": f"No chunks found for {source}."}]


@tool(parse_docstring=True)
def web_search(query: str) -> list[dict]:
    """Search the public web.

    Use for current events, weather, news, prices, versions, and general knowledge, or when the
    user's documents had no relevant answer. Returns titles, urls, snippets, and the text of the top
    pages.

    Args:
        query: Search query.
    """
    last_error = None
    for backend in ("duckduckgo", "bing", "auto"):
        try:
            hits = DDGS(timeout=15).text(query, max_results=3, backend=backend)
        except Exception as e:  # engine blocked or timed out: try the next one
            last_error = e
            continue
        if hits:
            results = [{"title": h["title"], "url": h["href"], "snippet": h["body"]} for h in hits]
            # Read the top pages ourselves: snippets alone rarely hold the answer.
            for r in results[:2]:
                text = _page_text(r["url"], PAGE_CHARS)
                if len(text) > 200 and not text.startswith("Could not fetch"):
                    r["content"] = text
            return results
    return [{"title": "Search failed", "url": "", "snippet": f"All engines failed: {last_error}"}]


@tool(parse_docstring=True)
def fetch_page(url: str) -> str:
    """Download a web page and return its text. Use on a url from web_search.

    Args:
        url: Full page url.
    """
    return _page_text(url, 3000)


@tool(parse_docstring=True)
def save_report(title: str, content: str) -> dict:
    """Save a finished report as a Markdown file. Needs the user's approval first.

    Call only after gathering information, and only once. Use when the user asks for a report,
    write-up, or document to keep.

    Args:
        title: Short report title.
        content: The complete report in Markdown.
    """
    # Human in the loop: execution stops here until the app resumes it with the user's decision.
    decision = interrupt({"tool": "save_report", "args": {"title": title, "content": content}})
    if not decision.get("approved"):
        return {"status": "rejected", "message": "The user rejected this action. It was not saved."}
    return write_report(decision.get("title") or title, decision.get("content") or content)


TOOLS = [search_documents, list_documents, get_chunks, web_search, fetch_page, save_report]
GATED_TOOLS = {"save_report"}  # tools that pause for approval
