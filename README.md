# AI Model Benchmarker

A hands-on project for comparing large language models across hosted APIs (OpenAI, Anthropic, OpenRouter) and local models (Ollama) on quality, latency, and cost — built over a structured 4-week plan.

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

For local models, install [Ollama](https://ollama.com) and pull a model, e.g. `ollama pull llama3`.

## Project Status

🚧 Week 1 — initial setup complete.
