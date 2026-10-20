# Agent demo   an elyaeval case study

An "energy analyst" tool-calling agent built with **LangGraph**, evaluated with
[elyaeval](https://github.com/FidaMarzouk/DeepEval-based-Testing-Framework-for-LLM-Applications) at two
levels: the **actions** it takes (which tools, which arguments, is the answer right) over HTTP, and its
whole **trajectory** (task completion, step efficiency) in-process   plus
deterministic guards against runaway agents. This folder is a **self-contained project**: copy it into
its own repository as-is.

**Stack** (everything runs locally by default)
- Agent: LangGraph state graph (model node ⇄ tool node), max-steps limit, output-token cap
- Model: `llama3.2:3b` via Ollama (`LLM_PROVIDER=groq` + `GROQ_API_KEY` to use Groq instead)
- Tools: deterministic calculations over `src/data/energy_data.json` (no database, no external API)
- Judge (evaluation only): `llama3.1` via Ollama, configured in `elyaeval.toml` (any provider works
  there)   a different model from the agent's

## The agent's tools

| Tool | Does |
|---|---|
| `get_technology_facts(technology)` | capacity-factor range + typical value, lifetime, lifecycle gCO2/kWh for solar PV, onshore/offshore wind, hydropower, geothermal, nuclear, natural gas, coal |
| `estimate_annual_generation(capacity_mw, capacity_factor)` | MWh per year = MW × CF × 8760 |
| `convert_energy(value, from_unit, to_unit)` | kWh, MWh, GWh, TWh, GJ |
| `estimate_co2_avoided(energy_mwh, replaced_technology, new_technology)` | tonnes of lifecycle CO2 avoided |

## Project layout

```
agent-demo/
├── src/
│   ├── config.py                          # settings from env / .env
│   ├── tools.py                           # the tools, traced as DeepEval tool spans
│   ├── agent.py                           # LangGraph graph + run_agent() (traced agent / llm spans)
│   ├── api.py                             # FastAPI: POST /chat, GET /health
│   └── data/energy_data.json              # the tools' data
├── requirements.txt                       # the agent's dependencies
│
├── elyaeval.toml                          # evaluation config: judge, thresholds, cache, budget, regression gate
├── goldens_agent_demo.jsonl               # 19 goldens with expected tools / answers (see below)
├── test_elyaeval_agent.py                 # action-level suite: calls the running API over HTTP
├── test_elyaeval_agent_trajectory.py      # trajectory suite: runs the agent in-process under tracing
├── requirements-eval.txt                  # agent deps + elyaeval, for running the suites locally
└── pipelinerun.yaml                       # CI: this project's run of elyaeval's Tekton pipeline
```

## 1. Run the agent

```bash
ollama pull llama3.2:3b           # the agent's model
ollama pull llama3.1              # evaluation judge

python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-eval.txt                  # agent + elyaeval
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

`POST /chat` with `{"message": "Convert 250 MWh to GWh."}` →
`{"message", "answer", "tool_calls": [{"name", "input_parameters", "output"}], "steps", "stopped",
"usage": {"input_tokens", "output_tokens"}, "latency_seconds"}`. Returning `tool_calls` is what lets an
evaluator score the agent's actions over HTTP.

| Setting (env or `.env`) | Default | |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `llama3.2:3b` | `groq` / `llama-3.3-70b-versatile` with `GROQ_API_KEY` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `MAX_STEPS` | `6` | model turns before the agent gives up (`stopped: "max_steps"`) |
| `MAX_OUTPUT_TOKENS` | `512` | per model turn   a small local model can otherwise run away |
| `LLM_TEMPERATURE` | `0` | |

## The goldens

`goldens_agent_demo.jsonl`: 19 requests, each with the tools the agent should call (`expected_tools`;
`[]` = it should call none), a reference answer computed from the same data the tools use, a
`max_tool_calls` budget, a category and a CI stage (cumulative).

| Category | What it checks |
|---|---|
| `single_tool` | one lookup / conversion / estimate |
| `multi_step` | chaining tools (look up a capacity factor, then estimate generation, then convert or compute CO2) |
| `no_tool`, `out_of_scope` | greetings and requests it can't serve: answer without calling tools |
| `tool_error` | an unknown technology or unit: report the error, don't invent a number |
| `adversarial` | asked to ignore its tools and make a number up |

| Stage | Goldens (cumulative) |
|---|---|
| `per_commit` | 3 |
| `merge` | 8 |
| `nightly` | 14 |
| `release` | 19 |

## 2. Evaluate locally

```bash
# action level, over HTTP (API running on :8001)
ELYAEVAL_CI_STAGE=per_commit pytest test_elyaeval_agent.py            # PowerShell: $env:ELYAEVAL_CI_STAGE="per_commit"; pytest ...

# trajectory level, in-process (needs Ollama; slower: the judge reads the whole trace)
ELYAEVAL_CI_STAGE=per_commit pytest test_elyaeval_agent_trajectory.py
```

| Suite | Metrics |
|---|---|
| action | Tool Correctness (vs `expected_tools`), Argument Correctness, Correctness (G-Eval vs the reference answer) |
| trajectory | Task Completion, Step Efficiency (Plan Quality / Adherence suit agents that write an explicit plan; this one doesn't) |
| both | Tool Call Budget (≤ `max_tool_calls`), Repeated Tool Calls (no identical call twice)   deterministic, no judge |

Results land in `report/` (CSV, HTML, run metadata, budget ledger).

## 3. Evaluate in CI (Tekton)

CI runs the **action-level** suite against the **already-running** agent API, inside elyaeval's runner
image, using the shared `elyaeval` Pipeline (one-time cluster setup: elyaeval's CI guide,
`elyaeval/src/elyaeval/recipes/tekton/README.md` in the elyaeval repo). Per run, with the agent API
and Ollama running on your machine:

```powershell
$env:REPO_URL         = "https://github.com/<you>/<this-repo>.git"
$env:SUT_URL          = "http://host.docker.internal:8001"
$env:JUDGE_PROVIDER   = "ollama"
$env:JUDGE_MODEL_NAME = "llama3.1"
$env:JUDGE_BASE_URL   = "http://host.docker.internal:11434"
envsubst < pipelinerun.yaml | kubectl create -n elyaeval -f -
```

Results go to the `agent-demo` blob container; the run fails on metric thresholds, an exhausted budget
or a regression against the stored baseline (`promote-baseline: "true"` once to create it).
