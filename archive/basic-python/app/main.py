import json
from pathlib import Path
from typing import Literal

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.ingest import build_chunks
from pydantic import BaseModel

from app.agent import resume_agent_stream, run_agent, run_agent_stream
from app.rag import answer_question
from app.tools import save_report, web_search
from app.vectorstore import add_chunks, delete_document, list_documents, search

app = FastAPI(title="AI Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ollama-check")
def ollama_check():
    response = ollama.list()
    return {"models": [m.model for m in response.models]}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = UPLOAD_DIR / Path(file.filename).name
    path.write_bytes(await file.read())
    chunks = build_chunks(path)
    stored = add_chunks(chunks)
    return {"filename": path.name, "chunks_stored": stored}


class AskRequest(BaseModel):
    question: str
    k: int = 2


@app.post("/ask")
def ask(body: AskRequest):
    return answer_question(body.question, body.k)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []  # earlier messages of this chat, oldest first


def history_of(body: AgentRequest) -> list[dict]:
    return [m.model_dump() for m in body.history]


@app.post("/agent")
def agent(body: AgentRequest):
    return run_agent(body.question, history_of(body))


@app.post("/agent/stream")
def agent_stream(body: AgentRequest):
    def event_source():
        for event in run_agent_stream(body.question, history_of(body)):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


class ApprovalRequest(BaseModel):
    id: str
    approved: bool
    title: str | None = None  # optional edits made in the approval card
    content: str | None = None


@app.post("/agent/approve")
def agent_approve(body: ApprovalRequest):
    edits = {"title": body.title, "content": body.content}

    def event_source():
        for event in resume_agent_stream(body.id, body.approved, edits):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


class ReportRequest(BaseModel):
    title: str
    content: str


@app.post("/reports")
def create_report(body: ReportRequest):
    """Save a report directly (the 'Save as report' button). The user already approved it in the UI."""
    if not body.title.strip() or not body.content.strip():
        raise HTTPException(status_code=400, detail="Title and content are required")
    return save_report(body.title.strip(), body.content)


@app.get("/reports")
def reports():
    folder = Path("data/reports")
    names = sorted((p.name for p in folder.glob("*.md")), reverse=True) if folder.exists() else []
    return {"reports": names}


@app.delete("/reports/{name}")
def delete_report(name: str):
    path = Path("data/reports") / Path(name).name  # .name blocks '../' style names
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    path.unlink()
    return {"deleted": path.name}


@app.get("/reports/{name}")
def report_file(name: str):
    path = Path("data/reports") / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.delete("/documents/{name}")
def delete_document_endpoint(name: str):
    """Delete the uploaded file and all its chunks together."""
    if Path(name).name != name:  # blocks '../' style names
        raise HTTPException(status_code=400, detail="Invalid file name")
    chunks_removed = delete_document(name)
    file = UPLOAD_DIR / name
    file_removed = file.is_file()
    if file_removed:
        file.unlink()
    if not chunks_removed and not file_removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": name, "chunks_removed": chunks_removed, "file_removed": file_removed}


@app.get("/web-search")
def web_search_endpoint(q: str, n: int = 3):
    return {"query": q, "results": web_search(q, n)}


@app.get("/search")
def search_docs(q: str, k: int = 2):
    return {"query": q, "results": search(q, k)}
