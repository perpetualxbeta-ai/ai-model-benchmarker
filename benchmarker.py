"""Async benchmark runner: queries x model tiers -> rich tables + CSV.

Usage:
    python benchmarker.py                      # all tiers, all 20 queries, 1 run each
    python benchmarker.py --tiers mid lightweight --runs 3
    python benchmarker.py --categories code_generation --limit 2
    python benchmarker.py --dry-run            # mocked responses, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm
import pandas as pd
import yaml
from dotenv import load_dotenv
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from metrics import (
    Billing,
    Pricing,
    StreamResult,
    billed_cost_usd,
    list_cost_usd,
    stream_completion,
    summarize,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "models.yaml"
DEFAULT_QUERIES = ROOT / "queries.json"
DEFAULT_OUTPUT_DIR = ROOT / "results"
TIER_ORDER = ("frontier", "mid", "lightweight")

# Transient failures worth retrying; anything else is recorded and skipped.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
    Timeout,
    APIConnectionError,
)

console = Console()


# --------------------------------------------------------------------------- config


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    name: str
    max_concurrency: int
    env_key: str | None = None
    env_base_url: str | None = None
    default_base_url: str | None = None


@dataclass(slots=True, frozen=True)
class ModelConfig:
    tier: str
    model: str
    provider: str
    pricing: Pricing
    params: tuple[tuple[str, Any], ...] = ()


@dataclass(slots=True, frozen=True)
class Query:
    id: str
    category: str
    prompt: str
    system: str


@dataclass(slots=True)
class RunRecord:
    """One row per request in results/benchmark_runs.csv."""

    run_id: str
    timestamp: str
    tier: str
    model: str
    provider: str
    query_id: str
    category: str
    repeat: int
    ttft_s: float | None
    latency_s: float | None
    output_tokens_per_s: float | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    usage_estimated: bool
    cost_billed_usd: float
    cost_list_usd: float
    retries: int
    finish_reason: str | None
    error: str | None


def load_providers(raw: dict[str, Any]) -> dict[str, ProviderConfig]:
    return {
        name: ProviderConfig(
            name=name,
            max_concurrency=int(cfg.get("max_concurrency", 2)),
            env_key=cfg.get("env_key"),
            env_base_url=cfg.get("env_base_url"),
            default_base_url=cfg.get("default_base_url"),
        )
        for name, cfg in raw["providers"].items()
    }


def _model_from_raw(tier: str, raw: dict[str, Any]) -> ModelConfig:
    billing: Billing = raw.get("billing", "paid")
    if billing not in ("paid", "free", "local"):
        raise ValueError(f"{raw['model']}: billing must be paid, free or local, got {billing!r}")
    return ModelConfig(
        tier=tier,
        model=raw["model"],
        provider=raw["provider"],
        pricing=Pricing(
            input_per_1m=float(raw.get("input_per_1m", 0.0)),
            output_per_1m=float(raw.get("output_per_1m", 0.0)),
            billing=billing,
        ),
        params=tuple(sorted((raw.get("params") or {}).items())),
    )


def load_tier_candidates(raw: dict[str, Any], tiers: Sequence[str]) -> dict[str, list[ModelConfig]]:
    """Primary then fallback model for each requested tier."""
    candidates: dict[str, list[ModelConfig]] = {}
    for tier in tiers:
        spec = raw["tiers"].get(tier)
        if spec is None:
            raise SystemExit(f"Unknown tier {tier!r}; config has {', '.join(raw['tiers'])}")
        candidates[tier] = [_model_from_raw(tier, spec[slot]) for slot in ("primary", "fallback") if spec.get(slot)]
    return candidates


def load_queries(path: Path, categories: Sequence[str] | None, limit: int | None) -> list[Query]:
    data = json.loads(path.read_text(encoding="utf-8"))
    default_system = data.get("default_system_prompt", "")
    queries = [
        Query(id=q["id"], category=q["category"], prompt=q["prompt"], system=q.get("system", default_system))
        for q in data["queries"]
        if not categories or q["category"] in categories
    ]
    if limit is not None:
        # Limit applies per category so a quick run still covers every category.
        per_cat: dict[str, int] = {}
        kept: list[Query] = []
        for q in queries:
            if per_cat.get(q.category, 0) < limit:
                kept.append(q)
                per_cat[q.category] = per_cat.get(q.category, 0) + 1
        queries = kept
    return queries


# --------------------------------------------------------------------------- credentials


def provider_kwargs(provider: ProviderConfig) -> dict[str, Any] | None:
    """litellm kwargs for a provider, or None if its credential isn't set.

    litellm reads GEMINI_API_KEY and OPENROUTER_API_KEY from the environment
    itself; we pass them explicitly so a missing key is detected up front
    instead of surfacing as an auth error mid-run.
    """
    kwargs: dict[str, Any] = {}
    if provider.env_key:
        key = os.getenv(provider.env_key, "").strip()
        if not key or key.startswith("your_"):
            return None
        kwargs["api_key"] = key
    if provider.env_base_url:
        base = os.getenv(provider.env_base_url, "").strip() or provider.default_base_url
        if not base:
            return None
        kwargs["api_base"] = base.rstrip("/")
    return kwargs


def redact(message: str, providers: dict[str, ProviderConfig]) -> str:
    """Strip any configured API key values from an error message."""
    for provider in providers.values():
        if provider.env_key:
            secret = os.getenv(provider.env_key, "")
            if len(secret) >= 8:
                message = message.replace(secret, "***")
    return message


# --------------------------------------------------------------------------- execution


@dataclass(slots=True)
class Settings:
    runs: int
    max_tokens: int
    timeout_s: float
    max_retries: int
    seed: int | None
    dry_run: bool
    sequential: bool


def build_messages(query: Query) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": query.prompt}]
    if query.system:
        messages.insert(0, {"role": "system", "content": query.system})
    return messages


def request_kwargs(model: ModelConfig, base: dict[str, Any], settings: Settings, query_id: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        **base,
        "temperature": 0,
        "max_tokens": settings.max_tokens,
        "timeout": settings.timeout_s,
        **dict(model.params),  # per-model overrides from config win
    }
    # A null in config means "don't send this param" (e.g. temperature for Gemini 3+).
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if settings.seed is not None:
        kwargs["seed"] = settings.seed
    if settings.dry_run:
        kwargs["mock_response"] = f"[dry-run] Mocked answer from {model.model} for {query_id}. " * 4
    return kwargs


async def timed_request(
    model: ModelConfig, messages: list[dict[str, str]], kwargs: dict[str, Any], max_retries: int
) -> tuple[StreamResult, int]:
    """Stream one completion, retrying transient errors.

    Only the successful attempt is timed; backoff sleeps happen outside the
    measured window and are reported as a retry count instead.
    """
    attempt = 0
    while True:
        try:
            return await stream_completion(model.model, messages, **kwargs), attempt
        except RETRYABLE_ERRORS:
            if attempt >= max_retries:
                raise
            attempt += 1
            await asyncio.sleep(min(30.0, 2.0**attempt) * (1 + random.uniform(0, 0.1)))


async def warm_up(model: ModelConfig, base: dict[str, Any], settings: Settings) -> str | None:
    """Discarded request that loads the model (Ollama) and checks credentials. Returns an error or None."""
    probe = Query(id="warmup", category="warmup", prompt="Reply with the single word: ready", system="")
    kwargs = request_kwargs(model, base, settings, "warmup") | {"max_tokens": 16}
    try:
        await timed_request(model, build_messages(probe), kwargs, max_retries=1)
    except Exception as exc:  # any failure means "use the fallback"
        return f"{type(exc).__name__}: {exc}"
    return None


async def resolve_models(
    candidates: dict[str, list[ModelConfig]],
    providers: dict[str, ProviderConfig],
    settings: Settings,
) -> list[tuple[ModelConfig, dict[str, Any]]]:
    """Pick the first usable model per tier: credential present and warm-up succeeds."""
    chosen: list[tuple[ModelConfig, dict[str, Any]]] = []
    for tier, models in candidates.items():
        for model in models:
            base = {} if settings.dry_run else provider_kwargs(providers[model.provider])
            if base is None:
                env = providers[model.provider].env_key or providers[model.provider].env_base_url
                console.print(f"  [yellow]•[/] {tier}: skipping {model.model} ({env} not set)")
                continue
            with console.status(f"Warming up {model.model}…"):
                error = await warm_up(model, base, settings)
            if error:
                console.print(f"  [yellow]•[/] {tier}: {model.model} failed warm-up: {redact(error, providers)[:200]}")
                continue
            console.print(f"  [green]✓[/] {tier}: {model.model}")
            chosen.append((model, base))
            break
        else:
            console.print(f"  [red]✗[/] {tier}: no usable model, tier skipped")
    return chosen


async def run_benchmark(
    selected: list[tuple[ModelConfig, dict[str, Any]]],
    queries: list[Query],
    providers: dict[str, ProviderConfig],
    settings: Settings,
    run_id: str,
) -> tuple[list[RunRecord], list[dict[str, Any]]]:
    """Run every query against every selected model, `runs` times each."""
    global_limit = asyncio.Semaphore(1) if settings.sequential else None
    limits = {name: asyncio.Semaphore(p.max_concurrency) for name, p in providers.items()}
    records: list[RunRecord] = []
    responses: list[dict[str, Any]] = []

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    async def one(model: ModelConfig, base: dict[str, Any], query: Query, repeat: int, task_id: Any) -> None:
        kwargs = request_kwargs(model, base, settings, query.id)
        async with limits[model.provider]:
            if global_limit:
                await global_limit.acquire()
            try:
                result, retries = await timed_request(model, build_messages(query), kwargs, settings.max_retries)
                m = result.metrics
                record = RunRecord(
                    run_id=run_id,
                    timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                    tier=model.tier,
                    model=model.model,
                    provider=model.provider,
                    query_id=query.id,
                    category=query.category,
                    repeat=repeat,
                    ttft_s=m.ttft_s,
                    latency_s=m.latency_s,
                    output_tokens_per_s=m.output_tokens_per_s,
                    prompt_tokens=m.prompt_tokens,
                    completion_tokens=m.completion_tokens,
                    reasoning_tokens=m.reasoning_tokens,
                    total_tokens=m.total_tokens,
                    usage_estimated=m.usage_estimated,
                    cost_billed_usd=billed_cost_usd(m.prompt_tokens, m.completion_tokens, model.pricing),
                    cost_list_usd=list_cost_usd(m.prompt_tokens, m.completion_tokens, model.pricing),
                    retries=retries,
                    finish_reason=result.finish_reason,
                    error=None,
                )
                responses.append(
                    {
                        "run_id": run_id,
                        "model": model.model,
                        "query_id": query.id,
                        "repeat": repeat,
                        "response": result.text,
                    }
                )
            except Exception as exc:  # one bad request must not abort the run
                record = RunRecord(
                    run_id=run_id,
                    timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                    tier=model.tier,
                    model=model.model,
                    provider=model.provider,
                    query_id=query.id,
                    category=query.category,
                    repeat=repeat,
                    ttft_s=None,
                    latency_s=None,
                    output_tokens_per_s=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_tokens=0,
                    total_tokens=0,
                    usage_estimated=False,
                    cost_billed_usd=0.0,
                    cost_list_usd=0.0,
                    retries=settings.max_retries if isinstance(exc, RETRYABLE_ERRORS) else 0,
                    finish_reason=None,
                    error=redact(f"{type(exc).__name__}: {exc}", providers)[:500],
                )
            finally:
                if global_limit:
                    global_limit.release()
        records.append(record)
        progress.advance(task_id)

    with progress:
        async with asyncio.TaskGroup() as tg:
            for model, base in selected:
                task_id = progress.add_task(f"{model.tier:<11} {model.model}", total=len(queries) * settings.runs)
                for repeat in range(1, settings.runs + 1):
                    for query in queries:
                        tg.create_task(one(model, base, query, repeat, task_id))
    return records, responses


# --------------------------------------------------------------------------- reporting


def _fmt(value: Any, spec: str, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:{spec}}{suffix}"


def _fmt_cost(value: float, billing: str) -> str:
    if billing == "local":
        return "$0 local"
    if billing == "free":
        return "$0 free"
    return _money(value)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts (typed, unlike itertuples)."""
    return [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]


def _short(model: str) -> str:
    """Drop the litellm provider prefix for display (gemini/x -> x)."""
    return model.split("/", 1)[1] if "/" in model else model


def _money(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def render_summary(summary: pd.DataFrame) -> None:
    perf = Table(title="Performance", header_style="bold cyan")
    perf.add_column("Tier")
    perf.add_column("Model", no_wrap=True)
    for col in ("OK/Err", "TTFT p50", "TTFT p95", "Lat. mean", "Lat. p95", "Tok/s"):
        perf.add_column(col, justify="right")

    cost = Table(title="Tokens & cost", header_style="bold cyan")
    cost.add_column("Tier")
    cost.add_column("Model", no_wrap=True)
    for col in ("Avg in", "Avg out", "Total tok", "Cost/1k billed", "Cost/1k list"):
        cost.add_column(col, justify="right")

    for row in _records(summary):
        name = _short(row["model"]) + (" *" if row["usage_estimated"] else "")
        perf.add_row(
            row["tier"],
            name,
            f"{row['requests'] - row['errors']}/{row['errors']}",
            _fmt(row["ttft_p50_s"], ".2f", "s"),
            _fmt(row["ttft_p95_s"], ".2f", "s"),
            _fmt(row["latency_mean_s"], ".2f", "s"),
            _fmt(row["latency_p95_s"], ".2f", "s"),
            _fmt(row["output_tokens_per_s"], ".0f"),
        )
        cost.add_row(
            row["tier"],
            name,
            _fmt(row["avg_prompt_tokens"], ".0f"),
            _fmt(row["avg_completion_tokens"], ".0f"),
            f"{row['total_tokens']:,}",
            _fmt_cost(row["cost_per_1k_billed_usd"], row["billing"]),
            _money(row["cost_per_1k_list_usd"]),
        )
    console.print(perf)
    console.print(cost)
    if summary["usage_estimated"].any():
        console.print("[dim]* token counts estimated with a tokenizer; the provider didn't return usage.[/]")


def render_by_category(runs: pd.DataFrame) -> None:
    ok = runs[runs["error"].isna()]
    if ok.empty:
        return
    pivot = ok.pivot_table(index="model", columns="category", values="latency_s", aggfunc="mean")
    ttft = ok.pivot_table(index="model", columns="category", values="ttft_s", aggfunc="mean")
    table = Table(title="Mean latency by category · TTFT in brackets", header_style="bold cyan")
    table.add_column("Model", no_wrap=True)
    for cat in pivot.columns:
        table.add_column(str(cat).split("_")[0], justify="right")
    for model in pivot.index:
        table.add_row(
            _short(str(model)),
            *[
                f"{_fmt(pivot.at[model, cat], '.2f', 's')} ({_fmt(ttft.at[model, cat], '.2f', 's')})"
                for cat in pivot.columns
            ],
        )
    console.print(table)


def render_errors(runs: pd.DataFrame) -> None:
    failed = runs[runs["error"].notna()]
    if failed.empty:
        return
    table = Table(title=f"Errors ({len(failed)})", header_style="bold red")
    table.add_column("Model", no_wrap=True)
    table.add_column("Query")
    table.add_column("Error", overflow="fold")
    for row in _records(failed.head(10)):
        table.add_row(_short(row["model"]), row["query_id"], str(row["error"])[:160])
    console.print(table)


def write_outputs(
    runs: pd.DataFrame, summary: pd.DataFrame, responses: list[dict[str, Any]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "benchmark_results.csv",
        "runs": output_dir / "benchmark_runs.csv",
        "responses": output_dir / "responses.jsonl",
    }
    summary.round(6).to_csv(paths["summary"], index=False)
    runs.round(6).to_csv(paths["runs"], index=False)
    with paths["responses"].open("w", encoding="utf-8") as fh:
        for item in responses:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return paths


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLM tiers on TTFT, latency, tokens and cost.")
    parser.add_argument("--tiers", nargs="+", default=list(TIER_ORDER), choices=TIER_ORDER)
    parser.add_argument("--categories", nargs="+", help="Only run these query categories.")
    parser.add_argument("--limit", type=int, help="Max queries per category.")
    parser.add_argument("--runs", type=int, default=1, help="Repeat each query N times (default 1).")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Output token cap per request.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for transient errors.")
    parser.add_argument("--seed", type=int, help="Sampling seed, where the provider supports it.")
    parser.add_argument("--sequential", action="store_true", help="One request at a time for the cleanest latency.")
    parser.add_argument("--dry-run", action="store_true", help="Use mocked responses; no API calls or keys needed.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    # Unsupported params (e.g. seed on some providers) are dropped rather than raising.
    litellm.drop_params = True
    litellm.suppress_debug_info = True

    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    providers = load_providers(raw_config)
    ollama = providers.get("ollama")
    if ollama and ollama.env_base_url:
        # litellm's own Ollama default reads OLLAMA_API_BASE; mirror our variable into it.
        os.environ.setdefault("OLLAMA_API_BASE", os.getenv(ollama.env_base_url) or ollama.default_base_url or "")

    queries = load_queries(args.queries, args.categories, args.limit)
    if not queries:
        console.print("[red]No queries matched the filters.[/]")
        return 1

    settings = Settings(
        runs=args.runs,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        max_retries=args.retries,
        seed=args.seed,
        dry_run=args.dry_run,
        sequential=args.sequential,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    console.rule(f"[bold]AI Model Benchmarker[/] · run {run_id}" + (" · [yellow]DRY RUN[/]" if args.dry_run else ""))
    console.print(f"{len(queries)} queries x {args.runs} run(s) · pricing as of {raw_config.get('last_updated', '?')}")
    console.print("Resolving model tiers:")

    candidates = load_tier_candidates(raw_config, args.tiers)
    selected = await resolve_models(candidates, providers, settings)
    if not selected:
        console.print("[red]No models available. Check your .env keys and that Ollama is running.[/]")
        return 1

    records, responses = await run_benchmark(selected, queries, providers, settings, run_id)
    runs_df = pd.DataFrame([asdict(r) for r in records])
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    runs_df = runs_df.sort_values(
        ["tier", "query_id", "repeat"], key=lambda s: s.map(tier_rank) if s.name == "tier" else s
    ).reset_index(drop=True)
    summary_df = summarize(runs_df, {m.model: m.pricing for m, _ in selected})

    console.print()
    render_summary(summary_df)
    render_by_category(runs_df)
    render_errors(runs_df)

    paths = write_outputs(runs_df, summary_df, responses, args.output_dir)
    summary_path = paths["summary"]
    shown = summary_path.relative_to(ROOT) if summary_path.is_relative_to(ROOT) else summary_path
    console.print(f"\nSummary CSV: [bold]{shown}[/]")
    console.print(f"[dim]Per-request rows: {paths['runs'].name} · Raw responses: {paths['responses'].name}[/]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
