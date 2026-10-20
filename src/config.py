"""
Agent settings from environment variables (and a local .env).

Defaults run locally on Ollama. The agent's model is deliberately NOT the
evaluation judge's model (llama3.1, see elyaeval.toml).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = {"ollama": "llama3.2:3b", "groq": "llama-3.3-70b-versatile"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    groq_api_key: str | None
    temperature: float
    max_steps: int
    max_output_tokens: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    # spans exist for evaluation; don't print Confident AI notices per request
    os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "NO")
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in _DEFAULT_MODEL:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(_DEFAULT_MODEL)}, got {provider!r}")
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", _DEFAULT_MODEL[provider]),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        # model turns (each may call tools) before the agent gives up
        max_steps=int(os.getenv("MAX_STEPS", "6")),
        # hard cap per model turn: a small local model can otherwise run away
        # (observed: 41k tokens / 7 minutes on a one-line greeting)
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "512")),
    )
