"""
The agent: a small LangGraph graph.

    START -> agent -+-> tools -> agent ...    (the model asked for tools)
                    +-> nudge -> agent        (report requested but save_report not called: retry once)
                    +-> END                   (final answer)

The save_report tool pauses the graph with interrupt() until the user approves or rejects it
(see tools.py). The checkpointer keeps the paused run, so it can be resumed later.
"""
import re

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.models import chat_model
from app.tools import TOOLS

MAX_STEPS = 6  # model calls per run

# Conversation memory: the client sends earlier messages, we keep the last few and trim them.
# The latest assistant reply stays long, because follow-ups like "save it" refer to it.
MAX_HISTORY_MESSAGES = 6
LAST_REPLY_CHARS = 2500
OLD_MESSAGE_CHARS = 300

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


def asked_for_report(messages: list[AnyMessage]) -> bool:
    """Does the latest user message ask for a report, or say yes to a save offer?"""
    last = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    text = messages[last].content
    if REPORT_INTENT.search(text):
        return True
    previous = next((m.content for m in reversed(messages[:last]) if isinstance(m, AIMessage)), "")
    return bool(AFFIRMATIVE.match(text)) and "save" in previous.lower()


def missed_save(messages: list[AnyMessage]) -> bool:
    """True if a report was requested but save_report was never called."""
    called = any(isinstance(m, ToolMessage) and m.name == "save_report" for m in messages)
    return asked_for_report(messages) and not called


def build_messages(question: str, history: list[dict]) -> list[AnyMessage]:
    """System prompt, then the last few trimmed history messages, then the new question."""
    kept = [m for m in history if m["content"].strip()][-MAX_HISTORY_MESSAGES:]
    last_reply = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=-1)
    past: list[AnyMessage] = []
    for i, m in enumerate(kept):
        text = m["content"][: LAST_REPLY_CHARS if i == last_reply else OLD_MESSAGE_CHARS]
        past.append(AIMessage(text) if m["role"] == "assistant" else HumanMessage(text))
    return [SystemMessage(SYSTEM_PROMPT), *past, HumanMessage(question)]


class AgentState(MessagesState):
    nudged: bool  # the corrective retry is allowed once per run


llm = chat_model().bind_tools(TOOLS)


def call_model(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}


def nudge(state: AgentState):
    return {"messages": [HumanMessage(SAVE_NUDGE)], "nudged": True}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    if not state.get("nudged") and missed_save(state["messages"]):
        return "nudge"
    return END


_builder = StateGraph(AgentState)
_builder.add_node("agent", call_model)
_builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
_builder.add_node("nudge", nudge)
_builder.add_edge(START, "agent")
_builder.add_conditional_edges("agent", route, ["tools", "nudge", END])
_builder.add_edge("tools", "agent")
_builder.add_edge("nudge", "agent")

checkpointer = InMemorySaver()
graph = _builder.compile(checkpointer=checkpointer)
