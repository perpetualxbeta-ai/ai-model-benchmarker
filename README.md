# AI Model Benchmarker

A hands-on project for comparing large language models across hosted APIs (Gemini, OpenRouter) and local models (Ollama) on quality, latency, and cost — built over a structured 4-week plan.

## Goals

- Run the same prompts against multiple providers through one consistent interface
- Measure **quality** (accuracy / rubric scores), **latency** (time to first token, total time), and **cost** (tokens × price)
- Produce reproducible results and clear comparison reports

## Supported Providers

| Provider   | Env variable          | Notes                              |
|------------|-----------------------|------------------------------------|
| Google Gemini | `GEMINI_API_KEY` | Gemini model family (free tier via Google AI Studio) |
| OpenRouter | `OPENROUTER_API_KEY`  | Access to many open & closed models|
| Ollama     | `OLLAMA_BASE_URL`     | Local models, default `http://localhost:11434` |

## 4-Week Plan

### Week 1 — Foundations & Provider Clients
- Set up project structure, virtual environment, and configuration (`.env`)
- Build a common client interface with adapters for Gemini, OpenRouter, and Ollama
- Send a single prompt to every provider and log raw responses, token usage, and timing
- **Deliverable:** one script that queries all providers with the same prompt

### Week 2 — Benchmark Datasets & Evaluation
- Assemble task sets: reasoning, coding, summarization, factual Q&A, instruction following
- Implement scoring: exact match, keyword/regex checks, and LLM-as-judge rubrics
- Store prompts, expected outputs, and results in a structured format (JSON/CSV)
- **Deliverable:** automated runner that scores every model on every task

### Week 3 — Performance, Cost & Reliability
- Measure latency (time to first token, total response time) and throughput
- Calculate cost per run from token counts and provider pricing
- Add retries, rate-limit handling, concurrency, and repeat runs for variance
- **Deliverable:** benchmark results with quality, latency, and cost per model

### Week 4 — Reporting & Polish
- Generate comparison tables and charts (leaderboard, cost vs. quality, latency distribution)
- Add a CLI for choosing models, tasks, and number of runs
- Write up findings, add tests, and document how to reproduce results
- **Deliverable:** final report and a reusable, documented benchmarking tool

## Getting Started

```bash
git clone https://github.com/perpetualxbeta-ai/ai-model-benchmarker.git
cd ai-model-benchmarker

python3 -m venv venv
source venv/bin/activate

cp .env.example .env   # then add your API keys
```

For local models, install [Ollama](https://ollama.com) and pull the lightweight-tier model: `ollama pull llama3.2:3b`.

## Running the Benchmark

```bash
pip install -r requirements.txt

python benchmarker.py --dry-run          # mocked responses, checks the pipeline without API calls
python benchmarker.py                    # all 3 tiers x 20 queries
python benchmarker.py --runs 3           # repeat each query for variance
python benchmarker.py --tiers mid lightweight --categories code_generation --limit 2
python benchmarker.py --sequential       # one request at a time for the cleanest latency numbers
```

Model tiers, fallbacks and per-1M-token prices live in `config/models.yaml`. Each tier uses its
primary model when the key is set and a warm-up request succeeds, otherwise its OpenRouter fallback.

| Tier        | Primary                         | Fallback                                     |
|-------------|---------------------------------|----------------------------------------------|
| Frontier    | `gemini/gemini-3.1-pro-preview` | `openrouter/anthropic/claude-sonnet-4.5`     |
| Mid         | `gemini/gemini-3.8-flash`       | `openrouter/meta-llama/llama-3.3-70b-instruct` |
| Lightweight | `ollama/llama3.2:3b`            | `openrouter/meta-llama/llama-3.2-3b-instruct` |

Outputs go to `results/`:

- `benchmark_results.csv` — one row per model: TTFT and latency (mean/p50/p95), tokens/sec, token usage, cost per 1,000 queries (billed and list price)
- `benchmark_runs.csv` — one row per request
- `responses.jsonl` — raw model outputs, for Week 2 scoring
- `SUMMARY.md` — findings report generated from the run (the only results file committed to git; dry runs write `SUMMARY.dry-run.md` instead)

Run the tests with `pytest`.

## Project Status

✅ Week 1 — async pipeline measuring TTFT, latency, tokens and cost across 3 model tiers.
