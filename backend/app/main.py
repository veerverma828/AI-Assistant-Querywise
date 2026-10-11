import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import store
from app.ingest import build_documents
from app.streaming import resume_events, run_events
from app.tools import REPORT_DIR, write_report

app = FastAPI(title="Querywise")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def sse(events: Iterator[dict]) -> StreamingResponse:
    """Server-Sent Events: one `data: {...}` line per event."""

    def lines():
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(lines(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- documents ----------


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    path = store.UPLOAD_DIR / Path(file.filename).name
    path.write_bytes(file.file.read())
    stored = store.add_documents(build_documents(path))
    return {"filename": path.name, "chunks_stored": stored}


@app.get("/documents")
def documents():
    return {"documents": store.list_documents()}


@app.delete("/documents/{name}")
def delete_document(name: str):
    """Delete the uploaded file and all its chunks together."""
    if Path(name).name != name:  # blocks '../' style names
        raise HTTPException(status_code=400, detail="Invalid file name")
    chunks_removed = store.delete_document(name)
    file = store.UPLOAD_DIR / name
    file_removed = file.is_file()
    if file_removed:
        file.unlink()
    if not chunks_removed and not file_removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": name, "chunks_removed": chunks_removed, "file_removed": file_removed}


# ---------- agent ----------


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []  # earlier messages of this chat, oldest first


@app.post("/agent/stream")
def agent_stream(body: AgentRequest):
    return sse(run_events(body.question, [m.model_dump() for m in body.history]))


class ApprovalRequest(BaseModel):
    id: str
    approved: bool
    title: str | None = None  # optional edits made in the approval card
    content: str | None = None


@app.post("/agent/approve")
def agent_approve(body: ApprovalRequest):
    edits = {"title": body.title, "content": body.content}
    return sse(resume_events(body.id, body.approved, edits))


# ---------- reports ----------


class ReportRequest(BaseModel):
    title: str
    content: str


@app.post("/reports")
def create_report(body: ReportRequest):
    """Save a report directly (the 'Save as report' button). The user already approved it in the UI."""
    if not body.title.strip() or not body.content.strip():
        raise HTTPException(status_code=400, detail="Title and content are required")
    return write_report(body.title.strip(), body.content)


@app.get("/reports")
def reports():
    names = sorted((p.name for p in REPORT_DIR.glob("*.md")), reverse=True) if REPORT_DIR.exists() else []
    return {"reports": names}


@app.get("/reports/{name}")
def report_file(name: str):
    path = REPORT_DIR / Path(name).name  # .name blocks '../' style names
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@app.delete("/reports/{name}")
def delete_report(name: str):
    path = REPORT_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    path.unlink()
    return {"deleted": path.name}
