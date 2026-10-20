"""
The energy-analyst agent: a LangGraph state graph alternating between the
model (which decides which tools to call) and a tool node, until the model
answers or MAX_STEPS model turns are used.

    agent ──(tool calls?)──► tools ──► agent ──► ... ──► END

Tracing for evaluation (DeepEval @observe, no judge or metrics here):
  run_agent        type="agent"  the whole run + update_current_trace(...)
  call_model       type="llm"    every model turn
  <tool functions> type="tool"   every tool execution (see src/tools.py)

Tools run sequentially in the graph's own node (not LangGraph's prebuilt
ToolNode, which may execute tool calls on other threads), so every tool span
stays inside the agent's trace.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Optional

from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

from .config import get_settings
from .tools import LANGCHAIN_TOOLS, TOOL_FUNCTIONS

SYSTEM_PROMPT = (
    "You are an energy analyst assistant. You can look up facts about electricity generation "
    "technologies (capacity factors, lifetimes, lifecycle CO2 emissions), estimate a plant's annual "
    "generation, convert energy units (kWh, MWh, GWh, TWh, GJ) and estimate CO2 avoided when one "
    "technology replaces another. Always use the tools for facts and numbers instead of guessing, "
    "and base your answer on their results. When a calculation needs a capacity factor that the user "
    "did not give, first look up the technology's typical_capacity_factor with get_technology_facts. If a tool returns an error, explain it. For greetings or "
    "questions about what you can do, answer briefly without calling tools. If a request is outside "
    "these capabilities, say so instead of calling tools. Keep the final answer short and include the "
    "numbers with their units."
)


@dataclass
class ToolCallRecord:
    name: str
    input_parameters: dict[str, Any]
    output: Any


@dataclass
class AgentResult:
    message: str
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    stopped: Optional[str] = None  # "max_steps" if the agent ran out of turns
    usage: dict[str, Optional[int]] = field(default_factory=lambda: {"input_tokens": None, "output_tokens": None})
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _model():
    s = get_settings()
    if s.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not s.groq_api_key:
            raise RuntimeError("LLM_PROVIDER=groq needs GROQ_API_KEY")
        llm = ChatGroq(model=s.llm_model, api_key=s.groq_api_key, temperature=s.temperature,
                       max_tokens=s.max_output_tokens)
    else:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=s.temperature,
                         num_predict=s.max_output_tokens)
    return llm.bind_tools(LANGCHAIN_TOOLS)


@observe(type="llm")
def call_model(messages: list[BaseMessage]) -> AIMessage:
    return _model().invoke(messages)


def _agent_node(state: MessagesState) -> dict:
    return {"messages": [call_model(state["messages"])]}


def _tools_node(state: MessagesState) -> dict:
    last = state["messages"][-1]
    results = []
    for call in getattr(last, "tool_calls", None) or []:
        fn = TOOL_FUNCTIONS.get(call["name"])
        try:
            output = fn(**call["args"]) if fn else {"error": f"unknown tool '{call['name']}'"}
        except TypeError as exc:  # wrong/missing arguments from the model
            output = {"error": f"bad arguments for {call['name']}: {exc}"}
        results.append(ToolMessage(content=json.dumps(output), tool_call_id=call["id"], name=call["name"]))
    return {"messages": results}


def _route(state: MessagesState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


@lru_cache(maxsize=1)
def graph():
    builder = StateGraph(MessagesState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def _collect(messages: list[BaseMessage]) -> tuple[list[ToolCallRecord], int, dict[str, Optional[int]]]:
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    calls, steps, tokens_in, tokens_out = [], 0, 0, 0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        steps += 1
        usage = getattr(m, "usage_metadata", None) or {}
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)
        for tc in m.tool_calls or []:
            raw = outputs.get(tc["id"])
            try:
                output = json.loads(raw) if raw is not None else None
            except (TypeError, json.JSONDecodeError):
                output = raw
            calls.append(ToolCallRecord(name=tc["name"], input_parameters=tc["args"], output=output))
    usage = {"input_tokens": tokens_in or None, "output_tokens": tokens_out or None}
    return calls, steps, usage


@observe(type="agent")
def run_agent(message: str) -> AgentResult:
    """Answer one user message: what POST /chat returns, and the traced
    entrypoint the trajectory evaluation calls in-process."""
    started = time.perf_counter()
    max_steps = get_settings().max_steps
    inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]}
    stopped = None
    messages: list[BaseMessage] = []
    try:
        # one step = a model turn + its tool calls = 2 graph super-steps
        for state in graph().stream(inputs, config={"recursion_limit": 2 * max_steps + 1}, stream_mode="values"):
            messages = state["messages"]
        last = messages[-1] if messages else None
        answer = last.content if isinstance(last, AIMessage) and not last.tool_calls else ""
    except GraphRecursionError:
        stopped = "max_steps"
        answer = f"I could not finish this request within {max_steps} steps."

    calls, steps, usage = _collect(messages)
    update_current_trace(
        input=message,
        output=answer,
        tools_called=[ToolCall(name=c.name, input_parameters=c.input_parameters, output=c.output) for c in calls],
    )
    return AgentResult(
        message=message,
        answer=answer,
        tool_calls=calls,
        steps=steps,
        stopped=stopped,
        usage=usage,
        latency_seconds=time.perf_counter() - started,
    )
