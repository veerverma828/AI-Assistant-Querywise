import json
import re
import uuid

import ollama

from app.tools import (
    fetch_page,
    get_chunks,
    list_documents,
    save_report,
    search_documents,
    web_search,
)

AGENT_MODEL = "qwen2.5:7b"
MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 4000

# Conversation memory: the client sends earlier messages, we keep the last few and trim them.
# The latest assistant reply stays long, because follow-ups like "save it" refer to it.
MAX_HISTORY_MESSAGES = 6
LAST_REPLY_CHARS = 2500
OLD_MESSAGE_CHARS = 300

# Tools with side effects: the agent pauses and waits for the user's approval before running them.
GATED_TOOLS = {"save_report"}
# Paused runs waiting for a decision: approval_id -> {messages, calls, step}. In memory only.
PENDING: dict[str, dict] = {}

SYSTEM_PROMPT = (
    "You are a research assistant with tools. Read each tool's description and choose the right one yourself. "
    "If a tool reports no relevant result, try another tool that could help. "
    "Search snippets are short and rarely contain the answer. "
    "If the question needs a specific value (temperature, price, score, version, date), "
    "you MUST call fetch_page on the most relevant url before answering. "
    "Never tell the user to visit a website themselves: read it yourself. "
    "Call one tool at a time. Never invent facts. "
    "Always keep names from the user's question (cities, people, products) in your search queries. "
    "If a tool returns an error or empty result, retry once with different wording before giving up. "
    "When you have enough information, reply with a short final answer and name your sources "
    "(file name and page, or url). "
    "If the user asks for a report, write-up, or document (or says yes or save it after a report was offered): "
    "first gather what you need with the other tools (skip this if the content is already in the conversation), "
    "then call save_report ONCE with a clear title and the full report in Markdown "
    "(headings, bullet points, sources at the end). Do NOT ask the user in text whether to save: "
    "call save_report directly, the app itself will ask the user to approve. "
    "After the report is saved, reply with ONE short sentence confirming the file name: do not repeat the report. "
    "If the user rejects it, do not call save_report again unless they ask for changes."
)

# Schemas: what the model reads to know a tool exists and how to call it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the user's own uploaded files (resume, PDFs, notes, reports). "
                "Use for any question about the user, their background, skills, projects, or content "
                "that could be in their files, even if the question does not mention documents. "
                "Not for news, weather, or public facts. "
                "Returns the 2 closest passages: read them, ignore any that do not help, "
                "and use web_search if none answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List which documents are stored, with chunk count and pages for each. Use when asked what documents or files are stored or uploaded.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunks",
            "description": "Show the stored text chunks of one document, in order. Use when asked to show or inspect a document's chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Exact file name, e.g. report.pdf"},
                    "limit": {"type": "integer", "description": "How many chunks to show (default 3)"},
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web. Use for current events, weather, news, prices, versions, "
                "and general knowledge, or when the user's documents had no relevant answer. "
                "Returns titles, urls, snippets, and the text of the top pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Download a web page and return its text. Use on a url from web_search.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full page url"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": (
                "Save a finished report as a Markdown file. Needs the user's approval first. "
                "Call only after gathering information, and only once. "
                "Use when the user asks for a report, write-up, or document to keep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short report title"},
                    "content": {"type": "string", "description": "The complete report in Markdown"},
                },
                "required": ["title", "content"],
            },
        },
    },
]

# Registry: tool name -> real Python function.
REGISTRY = {
    "search_documents": search_documents,
    "list_documents": list_documents,
    "get_chunks": get_chunks,
    "web_search": web_search,
    "fetch_page": fetch_page,
    "save_report": save_report,
}


def summarize(name: str, result) -> dict:
    """Short human summary of a tool result for the UI. The model still gets the full result."""
    if isinstance(result, dict):
        if "saved" in result:
            return {"text": "Report saved", "items": [result["saved"]]}
        return {"text": result.get("message", "Done"), "items": []}
    if isinstance(result, str):
        if name == "fetch_page" and not result.startswith("Could not fetch"):
            return {"text": f"Read {len(result):,} characters", "items": []}
        return {"text": result[:140], "items": []}
    if result and isinstance(result[0], dict) and "info" in result[0]:
        return {"text": result[0]["info"], "items": []}
    if name in ("search_documents", "get_chunks"):
        verb = "Found" if name == "search_documents" else "Read"
        noun = "passages" if name == "search_documents" else "chunks"
        return {
            "text": f"{verb} {len(result)} {noun}",
            "items": [f"{r['source']} · p.{r['page']}" for r in result],
        }
    if name == "list_documents":
        return {
            "text": f"{len(result)} documents stored",
            "items": [f"{r['source']} · {r['chunks']} chunks" for r in result],
        }
    if name == "web_search":
        read = sum(1 for r in result if "content" in r)
        return {
            "text": f"{len(result)} results, read {read} pages",
            "items": [r["title"][:70] for r in result],
        }
    return {"text": f"{len(result)} results", "items": []}


def run_tool(name: str, args: dict):
    func = REGISTRY.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        return func(**args)
    except Exception as e:  # model may send wrong arguments; tell it, don't crash
        return f"Tool error: {e}"


def _execute(messages: list, name: str, args: dict, result):
    """Append a tool result to the history and yield the events the UI shows."""
    result_text = json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS]
    yield {
        "type": "tool_result",
        "tool": name,
        "preview": result_text[:300],
        "summary": summarize(name, result),
    }
    messages.append({"role": "tool", "content": result_text, "tool_name": name})


def _run_calls(messages: list, calls: list, step: int):
    """
    Run tool calls in order. Returns True if it paused for approval (the caller must stop).
    A gated call is not executed: its state is stored in PENDING and the UI is asked to approve.
    """
    for i, call in enumerate(calls):
        name = call.function.name
        args = dict(call.function.arguments)
        if name in GATED_TOOLS:
            approval_id = uuid.uuid4().hex[:12]
            PENDING[approval_id] = {"messages": messages, "calls": calls[i:], "step": step}
            yield {"type": "approval_required", "id": approval_id, "tool": name, "args": args}
            return True
        yield {"type": "tool_call", "tool": name, "args": args}
        yield from _execute(messages, name, args, run_tool(name, args))
    return False


SAVE_NUDGE = (
    "You wrote the report but did not call the save_report tool. Call save_report now with a title "
    "and the full report as content. Do not ask for confirmation: the app asks the user to approve."
)


# "make me a report", "write a short 2-bullet report", "save it", "save this report" ...
REPORT_INTENT = re.compile(
    r"\b(?:make|create|write|generate|draft|prepare|produce|give|need|want)\b(?:\W+[\w-]+){0,3}?\W+(?:report|write-?up|document)\b"
    r"|\bsave\b.*\b(?:report|write-?up|summary|it|this|that)\b",
    re.IGNORECASE,
)
AFFIRMATIVE = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it)\b", re.IGNORECASE)


def _asked_for_report(messages: list) -> bool:
    """Does the latest user message ask for a report, or say yes to a save offer?"""
    last = max(i for i, m in enumerate(messages) if m["role"] == "user")
    text = messages[last]["content"]
    if REPORT_INTENT.search(text):
        return True
    previous = next((m["content"] for m in reversed(messages[:last]) if m["role"] == "assistant"), "")
    return bool(AFFIRMATIVE.match(text)) and "save" in previous.lower()


def _missed_save(messages: list) -> bool:
    """True if a report was requested but save_report was never called."""
    called = any(m.get("tool_name") == "save_report" for m in messages if m["role"] == "tool")
    return _asked_for_report(messages) and not called


def _clean_history(history: list[dict]) -> list[dict]:
    """Keep the last few messages. Only the latest assistant reply keeps its full length."""
    kept = [m for m in history if m["role"] in ("user", "assistant") and m["content"].strip()]
    kept = kept[-MAX_HISTORY_MESSAGES:]
    last_reply = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=-1)
    return [
        {"role": m["role"], "content": m["content"][: LAST_REPLY_CHARS if i == last_reply else OLD_MESSAGE_CHARS]}
        for i, m in enumerate(kept)
    ]


def _agent_loop(messages: list, first_step: int):
    """The think -> act loop. Stops at a final answer, at an approval pause, or at MAX_STEPS."""
    nudged = False
    for step in range(first_step, MAX_STEPS + 1):
        yield {"type": "thinking", "step": step}
        content = ""
        tool_calls = []
        for chunk in ollama.chat(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
            options={"temperature": 0.1},
            stream=True,
        ):
            piece = chunk.message
            if piece.content:
                content += piece.content
                yield {"type": "token", "text": piece.content}
            if piece.tool_calls:
                tool_calls.extend(piece.tool_calls)

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        if not tool_calls:
            if not nudged and step < MAX_STEPS and _missed_save(messages):
                nudged = True  # one correction only, so a stubborn model cannot loop forever
                yield {"type": "discard_text"}  # the UI drops the text it just streamed
                messages.append({"role": "user", "content": SAVE_NUDGE})
                continue
            yield {"type": "answer", "text": content}
            return

        paused = yield from _run_calls(messages, tool_calls, step)
        if paused:
            return

    yield {"type": "answer", "text": "Stopped: reached the step limit before finishing."}


def run_agent_stream(question: str, history: list[dict] | None = None):
    """Generator: yields one event dict per step so the UI can show live progress."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_clean_history(history or []),
        {"role": "user", "content": question},
    ]
    yield from _agent_loop(messages, 1)


def resume_agent_stream(approval_id: str, approved: bool, edits: dict | None = None):
    """Continue a paused run after the user approved or rejected the gated tool call."""
    state = PENDING.pop(approval_id, None)
    if state is None:
        yield {"type": "answer", "text": "This approval request has expired. Please ask again."}
        return

    messages, calls, step = state["messages"], state["calls"], state["step"]
    call = calls[0]
    name = call.function.name
    args = dict(call.function.arguments)
    args.update({k: v for k, v in (edits or {}).items() if v})  # user may have edited title/content

    yield {"type": "tool_call", "tool": name, "args": args}
    if approved:
        result = run_tool(name, args)
    else:
        result = {"status": "rejected", "message": "The user rejected this action. It was not saved."}
    yield from _execute(messages, name, args, result)

    paused = yield from _run_calls(messages, calls[1:], step)
    if paused:
        return
    yield from _agent_loop(messages, step + 1)


def run_agent(question: str, history: list[dict] | None = None) -> dict:
    """Non-streaming version: collect all events, return answer + trace."""
    trace, answer = [], ""
    for event in run_agent_stream(question, history):
        if event["type"] == "tool_call":
            trace.append({"tool": event["tool"], "args": event["args"]})
        elif event["type"] == "tool_result":
            trace[-1]["result_preview"] = event["preview"]
        elif event["type"] == "answer":
            answer = event["text"]
    return {"answer": answer, "trace": trace}
