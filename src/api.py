"""
HTTP API around the agent.

    uvicorn src.api:app --host 0.0.0.0 --port 8001

POST /chat {"message": "..."}
       -> {"message", "answer", "tool_calls": [{"name", "input_parameters", "output"}],
           "steps", "stopped", "usage": {"input_tokens", "output_tokens"}, "latency_seconds"}
GET  /health -> {"status": "ok", "llm": "<provider>:<model>", "max_steps": 6}

Returning the tool calls is what lets an evaluator score the agent's
actions over HTTP (elyaeval's tool_calls_from_json reads this shape).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import run_agent
from .config import get_settings

app = FastAPI(title="Energy analyst agent", description="A LangGraph tool-calling agent over energy data.")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ToolCallModel(BaseModel):
    name: str
    input_parameters: dict[str, Any]
    output: Any = None


class ChatResponse(BaseModel):
    message: str
    answer: str
    tool_calls: list[ToolCallModel]
    steps: int
    stopped: Optional[str] = None
    usage: dict[str, Optional[int]]
    latency_seconds: float


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(**run_agent(request.message).to_dict())


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "llm": f"{s.llm_provider}:{s.llm_model}", "max_steps": s.max_steps}
