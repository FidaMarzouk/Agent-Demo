"""
Action-level evaluation of the agent API (started from `elyaeval init --task-type agentic_action`).

Calls the RUNNING agent over HTTP (APP_URL, default http://localhost:8001)
for every golden up to the CI stage (ELYAEVAL_CI_STAGE, default nightly):

  - Tool Correctness      did it call the expected tools ([] = it should call none)
  - Argument Correctness  were the arguments right for the request
  - Correctness (G-Eval)  is the final answer right, vs the golden's reference
  - Tool Call Budget,     deterministic guards: no more calls than the golden's
    Repeated Tool Calls   max_tool_calls, no identical call repeated (loops)

Judge, thresholds, cache, budget and regression gate: elyaeval.toml. CI runs
this file through elyaeval's Tekton pipeline.
"""

import os
import time

import pytest
import requests
from deepeval.test_case import LLMTestCase
from elyaeval import evaluate_golden, load_standard_dataset, new_report_path
from elyaeval.core.agentic import (
    agent_action_metrics_for_golden,
    expected_tools_from_golden,
    tool_calls_from_json,
)
from elyaeval.core.dataset import golden_id
from elyaeval.core.metrics import qa_correctness_metrics
from elyaeval.core.run_contracts import AgenticRunResult, SUTUsage
from elyaeval.core.stages import resolve_stage

APP_URL = os.environ.get("APP_URL", "http://localhost:8001").rstrip("/")
REPORT_CSV = new_report_path("agent")

GOLDENS = load_standard_dataset(
    task_type="agentic_action",
    ci_stage=resolve_stage("nightly"),
    dataset_path="goldens_agent_demo.jsonl",
)


def run_app(message: str) -> AgenticRunResult:
    response = requests.post(f"{APP_URL}/chat", json={"message": message}, timeout=600)
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage") or {}
    return AgenticRunResult(
        actual_output=data["answer"],
        tools_called=tool_calls_from_json(data["tool_calls"]),
        usage=SUTUsage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_seconds=data.get("latency_seconds"),
        ),
    )


@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_agent(golden):
    started = time.perf_counter()
    run_result = run_app(golden.input)
    usage = run_result.usage or SUTUsage()
    if usage.latency_seconds is None:
        usage = usage._replace(latency_seconds=time.perf_counter() - started)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=run_result.actual_output,
        expected_output=golden.expected_output,
        tools_called=run_result.tools_called,
        expected_tools=expected_tools_from_golden(golden),
    )
    metrics = agent_action_metrics_for_golden(golden, action_metrics=qa_correctness_metrics)
    result = evaluate_golden(golden, test_case, metrics, csv_path=REPORT_CSV, sut_usage=usage)
    assert result.success, f"{golden_id(golden)} failed   see {REPORT_CSV} for per-metric scores"
