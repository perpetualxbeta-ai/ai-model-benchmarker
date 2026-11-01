"""Markdown findings report (results/SUMMARY.md) built from a benchmark run.

Findings are derived only from the measured numbers; nothing here is
hand-written per run, so the report can be regenerated after every run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True, frozen=True)
class RunMeta:
    """Context printed at the top of the report."""

    run_id: str
    generated_at: str
    query_count: int
    runs_per_query: int
    categories: tuple[str, ...]
    pricing_date: str
    dry_run: bool
    sequential: bool


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _short(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _secs(value: Any) -> str:
    return "—" if _is_missing(value) else f"{float(value):.2f}s"


def _num(value: Any, decimals: int = 0) -> str:
    return "—" if _is_missing(value) else f"{float(value):,.{decimals}f}"


def _usd(value: Any) -> str:
    if _is_missing(value):
        return "—"
    v = float(value)
    if v == 0:
        return "$0"
    return f"${v:,.4f}" if v < 1 else f"${v:,.2f}"


def _billed(row: dict[str, Any]) -> str:
    if row["billing"] == "local":
        return "$0 (local)"
    if row["billing"] == "free":
        return "$0 (free tier)"
    return _usd(row["cost_per_1k_billed_usd"])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> str:
    """Render a GitHub markdown table. `align` has one of l/r per column."""

    def cell(text: str) -> str:
        # A literal pipe would split the cell.
        return text.replace("|", "\\|")

    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _best(rows: list[dict[str, Any]], key: str, *, highest: bool = False) -> dict[str, Any] | None:
    candidates = [r for r in rows if not _is_missing(r[key])]
    if not candidates:
        return None
    return (max if highest else min)(candidates, key=lambda r: float(r[key]))


def _worst(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return _best(rows, key, highest=True)


def key_findings(summary: pd.DataFrame) -> list[str]:
    """Plain-language findings computed from the per-model summary."""
    rows = [r for r in _records(summary) if r["requests"] > r["errors"]]
    findings: list[str] = []
    if not rows:
        return ["No model completed any request, so there are no measurements to compare."]

    fast, slow = _best(rows, "ttft_p50_s"), _worst(rows, "ttft_p50_s")
    if fast and slow and fast is not slow:
        ratio = float(slow["ttft_p50_s"]) / float(fast["ttft_p50_s"]) if float(fast["ttft_p50_s"]) > 0 else None
        times = f" ({ratio:.1f}x faster)" if ratio else ""
        findings.append(
            f"**Fastest first token:** {_short(fast['model'])} ({fast['tier']}) at a median "
            f"{_secs(fast['ttft_p50_s'])}, vs {_secs(slow['ttft_p50_s'])} for {_short(slow['model'])}{times}."
        )

    quick, sluggish = _best(rows, "latency_mean_s"), _worst(rows, "latency_mean_s")
    if quick and sluggish and quick is not sluggish:
        findings.append(
            f"**Lowest total latency:** {_short(quick['model'])} averaged {_secs(quick['latency_mean_s'])} per "
            f"response; {_short(sluggish['model'])} was slowest at {_secs(sluggish['latency_mean_s'])} "
            f"(p95 {_secs(sluggish['latency_p95_s'])})."
        )

    tput = _best(rows, "output_tokens_per_s", highest=True)
    if tput:
        findings.append(
            f"**Highest throughput:** {_short(tput['model'])} generated "
            f"{_num(tput['output_tokens_per_s'])} output tokens/s after the first token."
        )

    priced = [r for r in rows if r["billing"] != "local"]
    cheap, dear = _best(priced, "cost_per_1k_list_usd"), _worst(priced, "cost_per_1k_list_usd")
    if cheap and dear and cheap is not dear and float(cheap["cost_per_1k_list_usd"]) > 0:
        multiple = float(dear["cost_per_1k_list_usd"]) / float(cheap["cost_per_1k_list_usd"])
        findings.append(
            f"**Cost spread:** at list price, 1,000 queries cost {_usd(dear['cost_per_1k_list_usd'])} on "
            f"{_short(dear['model'])} vs {_usd(cheap['cost_per_1k_list_usd'])} on {_short(cheap['model'])} "
            f"({multiple:.0f}x)."
        )
    free = [_short(r["model"]) for r in rows if r["billing"] in ("free", "local")]
    if free:
        findings.append(
            f"**$0 billed:** {', '.join(free)} cost nothing on this setup (free tier or local hardware); "
            "list-price costs are in the table for planning a paid deployment."
        )

    verbose = _worst(rows, "avg_completion_tokens")
    terse = _best(rows, "avg_completion_tokens")
    if verbose and terse and verbose is not terse and float(terse["avg_completion_tokens"]) > 0:
        findings.append(
            f"**Verbosity:** {_short(verbose['model'])} wrote {_num(verbose['avg_completion_tokens'])} output "
            f"tokens per answer on average vs {_num(terse['avg_completion_tokens'])} for "
            f"{_short(terse['model'])}, which drives both its latency and its cost."
        )

    failing = [r for r in _records(summary) if r["errors"]]
    if failing:
        parts = [f"{_short(r['model'])} {r['errors']}/{r['requests']}" for r in failing]
        findings.append(f"**Errors:** {', '.join(parts)} requests failed (see the errors section).")
    else:
        findings.append("**Reliability:** every request completed without errors.")
    return findings


def build_summary_markdown(summary: pd.DataFrame, runs: pd.DataFrame, meta: RunMeta) -> str:
    """Full SUMMARY.md contents for one run."""
    rows = _records(summary)
    out: list[str] = ["# Benchmark Summary", ""]

    if meta.dry_run:
        out += [
            "> **⚠️ DRY RUN — mocked responses.** No model was called, so the timings, token counts",
            "> and costs below only exercise the pipeline. They are not benchmark findings.",
            "",
        ]

    out += [
        f"- **Run:** `{meta.run_id}` · generated {meta.generated_at}",
        f"- **Workload:** {meta.query_count} queries x {meta.runs_per_query} run(s) "
        f"({', '.join(c.replace('_', ' ') for c in meta.categories)})",
        "- **Models:** " + ", ".join(f"{r['tier']}: `{r['model']}`" for r in rows),
        f"- **Pricing as of:** {meta.pricing_date} (`config/models.yaml`)",
        f"- **Mode:** {'sequential (one request at a time)' if meta.sequential else 'concurrent, capped per provider'}",
        "",
        "## Key findings",
        "",
        *[f"- {f}" for f in key_findings(summary)],
        "",
        "## Latency",
        "",
        markdown_table(
            ["Tier", "Model", "OK / total", "TTFT p50", "TTFT p95", "Latency mean", "Latency p95", "Output tok/s"],
            [
                [
                    r["tier"],
                    _short(r["model"]),
                    f"{r['requests'] - r['errors']} / {r['requests']}",
                    _secs(r["ttft_p50_s"]),
                    _secs(r["ttft_p95_s"]),
                    _secs(r["latency_mean_s"]),
                    _secs(r["latency_p95_s"]),
                    _num(r["output_tokens_per_s"]),
                ]
                for r in rows
            ],
            "llrrrrrr",
        ),
        "",
        "## Tokens and cost",
        "",
        markdown_table(
            ["Tier", "Model", "Avg input tok", "Avg output tok", "Total tokens", "Cost / 1k billed", "Cost / 1k list"],
            [
                [
                    r["tier"],
                    _short(r["model"]) + (" \\*" if r["usage_estimated"] else ""),
                    _num(r["avg_prompt_tokens"]),
                    _num(r["avg_completion_tokens"]),
                    _num(r["total_tokens"]),
                    _billed(r),
                    _usd(r["cost_per_1k_list_usd"]),
                ]
                for r in rows
            ],
            "llrrrrr",
        ),
        "",
        "Cost per 1,000 queries = (avg input tokens x input price + avg output tokens x output price) x 1,000.",
    ]
    if any(r["usage_estimated"] for r in rows):
        out.append("\\* Token counts estimated with a tokenizer because the provider didn't return usage.")

    ok = runs[runs["error"].isna()]
    if not ok.empty:
        latency = ok.pivot_table(index="model", columns="category", values="latency_s", aggfunc="mean")
        ttft = ok.pivot_table(index="model", columns="category", values="ttft_s", aggfunc="mean")
        cats = [str(c) for c in latency.columns]
        out += [
            "",
            "## Mean latency by category",
            "",
            "TTFT in brackets.",
            "",
            markdown_table(
                ["Model", *[c.replace("_", " ") for c in cats]],
                [
                    [
                        _short(str(model)),
                        *[f"{_secs(latency.at[model, c])} ({_secs(ttft.at[model, c])})" for c in cats],
                    ]
                    for model in latency.index
                ],
                "l" + "r" * len(cats),
            ),
        ]

    failed = runs[runs["error"].notna()]
    if not failed.empty:
        out += [
            "",
            "## Errors",
            "",
            markdown_table(
                ["Model", "Query", "Error"],
                [[_short(r["model"]), r["query_id"], str(r["error"])[:160]] for r in _records(failed.head(15))],
                "lll",
            ),
        ]

    out += [
        "",
        "## Caveats",
        "",
        "- Response quality isn't scored yet (Week 2); this run measures speed, tokens and cost only.",
        "- Cloud latency includes network time from the machine that ran the benchmark; "
        "local Ollama latency depends on its hardware.",
        "- Thinking models count reasoning tokens as output tokens, which raises their cost and latency.",
        "- Prices change often; check `config/models.yaml` against the providers' pricing pages.",
        "",
        "Raw data: `benchmark_results.csv` (per model), `benchmark_runs.csv` (per request), "
        "`responses.jsonl` (model outputs).",
        "",
    ]
    return "\n".join(out)


def write_summary(summary: pd.DataFrame, runs: pd.DataFrame, meta: RunMeta, output_dir: Path) -> Path:
    """Write the report. Dry runs go to a separate file so they never overwrite real findings."""
    name = "SUMMARY.dry-run.md" if meta.dry_run else "SUMMARY.md"
    path = output_dir / name
    path.write_text(build_summary_markdown(summary, runs, meta), encoding="utf-8")
    return path
