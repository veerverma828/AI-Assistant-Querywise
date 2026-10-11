"""
Turns a LangGraph run into the events the React UI understands:

    thinking, token, tool_call, tool_result, approval_required, discard_text, answer
"""
import json
import uuid
from collections.abc import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from app.agent import MAX_STEPS, build_messages, checkpointer, graph
from app.tools import GATED_TOOLS


def summarize(name: str, result) -> dict:
    """Short human summary of a tool result for the UI. The model still sees the full result."""
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
        verb, noun = ("Found", "passages") if name == "search_documents" else ("Read", "chunks")
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


def _parse(content: str):
    try:
        return json.loads(content)
    except ValueError:
        return content


def _config(thread_id: str) -> dict:
    # Each model call plus its tool call is two graph steps.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 2 * MAX_STEPS + 1}


def _stream_run(graph_input, thread_id: str, step: int, announce: bool) -> Iterator[dict]:
    """Run (or resume) the graph and yield UI events. `step` numbers the "thinking" events."""
    config = _config(thread_id)
    if announce:
        yield {"type": "thinking", "step": step}
    try:
        for mode, data in graph.stream(graph_input, config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield {"type": "token", "text": chunk.content}
                continue

            for node, update in data.items():
                if node == "__interrupt__":
                    # A gated tool wants approval: stop here, the run stays parked in the checkpointer.
                    pause = update[0].value
                    yield {"type": "approval_required", "id": thread_id, "tool": pause["tool"], "args": pause["args"]}
                    return
                if node == "agent":
                    for call in update["messages"][-1].tool_calls:
                        if call["name"] not in GATED_TOOLS:  # gated calls show up as the approval card
                            yield {"type": "tool_call", "tool": call["name"], "args": call["args"]}
                elif node == "tools":
                    for msg in update["messages"]:
                        yield {
                            "type": "tool_result",
                            "tool": msg.name,
                            "preview": msg.content[:300],
                            "summary": summarize(msg.name, _parse(msg.content)),
                        }
                    step += 1
                    yield {"type": "thinking", "step": step}
                elif node == "nudge":
                    yield {"type": "discard_text"}  # the UI drops the text it just streamed
                    step += 1
                    yield {"type": "thinking", "step": step}
    except GraphRecursionError:
        yield {"type": "answer", "text": "Stopped: reached the step limit before finishing."}
        return
    except Exception as e:  # for example Ollama is not running
        yield {"type": "answer", "text": f"The model call failed: {e}"}
        return

    final = graph.get_state(config).values["messages"][-1]
    yield {"type": "answer", "text": final.content}
    checkpointer.delete_thread(thread_id)


def run_events(question: str, history: list[dict]) -> Iterator[dict]:
    """Start a new run for a question."""
    thread_id = uuid.uuid4().hex[:12]
    graph_input = {"messages": build_messages(question, history), "nudged": False}
    yield from _stream_run(graph_input, thread_id, step=1, announce=True)


def resume_events(approval_id: str, approved: bool, edits: dict) -> Iterator[dict]:
    """Continue a paused run after the user approved or rejected the gated tool call."""
    state = graph.get_state(_config(approval_id))
    if not state.next:
        yield {"type": "answer", "text": "This approval request has expired. Please ask again."}
        return

    edits = {k: v for k, v in edits.items() if v}  # only what the user actually changed
    messages = state.values["messages"]
    for call in messages[-1].tool_calls:
        if call["name"] in GATED_TOOLS:
            yield {"type": "tool_call", "tool": call["name"], "args": {**call["args"], **edits}}

    model_calls = sum(isinstance(m, AIMessage) for m in messages)
    resume = Command(resume={"approved": approved, **edits})
    yield from _stream_run(resume, approval_id, step=model_calls, announce=False)
