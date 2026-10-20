"""
Trajectory-level evaluation of the agent (started from `elyaeval init --task-type agentic_trajectory`).

Runs the agent IN-PROCESS (src.agent.run_agent) under DeepEval tracing and
judges the whole run   the model turns and tool calls in its trace   rather
than only the final answer:

  - Task Completion, Step Efficiency       (judge)
  - Tool Call Budget, Repeated Tool Calls  (deterministic)

Plan Quality / Plan Adherence (the rest of elyaeval's agentic_trajectory
preset) judge an explicit plan; this ReAct-style agent doesn't write one, so
they are left out here (in a trial run they scored ~0 on every golden).

The agent's code only declares @observe(type="agent"/"llm"/"tool") spans; it
imports no judge. Needs Ollama (same environment as the API, see README).
Slower than the action-level suite: the judge reads the full trace.
"""

import pytest
from deepeval.dataset import Golden
from elyaeval import load_standard_dataset, new_report_path
from elyaeval.core.agentic import guard_metrics
from elyaeval.core.dataset import golden_id
from elyaeval.core.metrics import build_metric
from elyaeval.core.stages import resolve_stage
from elyaeval.reporting.tracing_report import (
    TracedRunner,
    append_traced_csv_rows,
    failed_results,
    traced_test_result_to_csv_rows,
)

from src.agent import run_agent

REPORT_CSV = new_report_path("agent_trajectory")

GOLDENS = load_standard_dataset(
    task_type="agentic_action",
    ci_stage=resolve_stage("nightly"),
    dataset_path="goldens_agent_demo.jsonl",
)


def call_sut(golden: Golden) -> None:
    run_agent(golden.input)


@pytest.mark.parametrize("golden", GOLDENS, ids=[golden_id(g) for g in GOLDENS])
def test_agent_trajectory(golden):
    trace_metrics = [build_metric("task_completion"), build_metric("step_efficiency")] + guard_metrics(golden)
    result = TracedRunner(golden, metrics=trace_metrics).run(call_sut)

    gid = golden_id(golden)
    priority = (golden.additional_metadata or {}).get("priority", "")
    rows = []
    for test_result in result.test_results:
        rows.extend(traced_test_result_to_csv_rows(gid, priority, test_result))
    append_traced_csv_rows(REPORT_CSV, rows)

    failed = failed_results(result)
    assert not failed, f"failed: {failed}   see {REPORT_CSV} for per-metric scores"
