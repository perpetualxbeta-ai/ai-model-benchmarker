# CLAUDE.md

Guidance for Claude (and other contributors) working in this repository.

## Project Scope

A benchmarking tool that runs the same prompts against multiple LLM providers and reports, per model:

- **Response quality** — exact match, keyword/regex checks, and LLM-as-judge rubric scores
- **Time-to-First-Token (TTFT)** — measured from request send to the first streamed content chunk
- **Latency** — total wall-clock time to the final chunk, plus derived throughput (output tokens/sec)
- **Token usage** — prompt, completion, and total tokens as reported by the provider
- **Cost per 1,000 queries** — `(avg prompt tokens × input price + avg completion tokens × output price) × 1,000`

Providers in scope: Google Gemini, OpenRouter, Ollama (local), with OpenAI and Anthropic adapters as optional extras (see `README.md` and `.env.example`).

Out of scope: training/fine-tuning, hosting a web UI, or storing results in a database. Output is files (CSV/JSON) and terminal reports.

## Tech Stack

- **Python 3.11+** (use `asyncio.TaskGroup`, `X | None` unions, built-in generics)
- **`litellm`** as the default unified client (`litellm.acompletion(..., stream=True)`); drop to a provider's direct SDK only when litellm lacks a feature or distorts timing
- **`pandas`** for aggregating runs and computing summary stats (mean, p50, p95, stdev)
- **`rich`** for console output: progress bars during runs, tables for leaderboards
- **`python-dotenv`** to load API keys from `.env` (never commit `.env`; keep `.env.example` in sync when adding a variable)

Pin dependencies in `requirements.txt` (or `pyproject.toml`). Tests use `pytest` + `pytest-asyncio`.

## Code Style

- **Clean, modular Python.** Small single-purpose modules; no logic in `__main__` beyond wiring.
- **Type hints everywhere**, including return types. Use `dataclass(slots=True)` or `TypedDict` for structured records (e.g. `RunResult`, `ModelConfig`). Code should pass `mypy --strict` and `ruff`.
- **Async by default.** All provider calls are `async`; run concurrent requests with `asyncio.TaskGroup` and cap concurrency with an `asyncio.Semaphore` per provider to respect rate limits.
- **Accurate timing:**
  - Use `time.perf_counter()` (never `time.time()`) for all latency measurements.
  - Start the timer immediately before the request is awaited; record TTFT on the first chunk that contains non-empty content (not role-only or empty deltas).
  - Always stream when measuring TTFT. Request usage in the stream (e.g. `stream_options={"include_usage": True}`) and fall back to a tokenizer estimate only if the provider omits it — flag estimated counts in the results.
  - Exclude client setup, retries' backoff sleeps, and scoring time from latency figures. Record retries as a separate count.
  - Do a warm-up request per model (discarded) before timed runs, especially for Ollama.
- **No hardcoded prices or model IDs in code.** Keep pricing and model lists in a config file (e.g. `config/models.yaml`) with a `last_updated` date.
- **Errors:** catch provider errors per request, record them in the result row (`error` field) and keep the run going; never let one failing model abort the benchmark.
- **Reproducibility:** fixed `temperature=0` (unless a task specifies otherwise), a `--seed` where supported, N repeat runs per prompt, and every result row tagged with timestamp, model, provider, and run ID.
- Docstrings on public functions; comments explain *why*, not *what*.

## Layout

```
benchmarker.py     # CLI + orchestration: tier resolution, async runner, rich tables, CSV export
metrics.py         # streamed request timing (TTFT, latency, throughput), token usage, cost math
queries.json       # 20 enterprise queries; `expected` blocks are reference data for Week 2 scoring
config/models.yaml # model tiers, fallbacks, per-1M-token prices, billing mode, per-model params
results/           # run outputs (gitignored)
tests/
```

As the project grows (scoring, reporting), split into a `benchmarker/` package with `scoring.py` and `report.py`.

## Workflow Notes

- Set up: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp .env.example .env`
- Check before committing: `pytest && ruff check . && mypy benchmarker.py metrics.py`
- Gemini 3+ models deprecate sampling params; their `params` in `config/models.yaml` set `temperature: null` so it isn't sent.
- Never print or log API keys; redact them in error messages.
- Add `results/` to `.gitignore` before committing any benchmark output.
- When changing metric definitions, update this file and the README so reported numbers stay comparable across runs.
