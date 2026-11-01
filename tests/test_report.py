from __future__ import annotations

from pathlib import Path

import pandas as pd

from report import RunMeta, build_summary_markdown, key_findings, markdown_table, write_summary


def _summary() -> pd.DataFrame:
    base = {
        "provider": "p",
        "ttft_mean_s": None,
        "ttft_p95_s": 1.0,
        "latency_p50_s": 1.0,
        "latency_p95_s": 4.0,
        "latency_stdev_s": 0.1,
        "avg_total_tokens": 300.0,
        "total_tokens": 6000,
        "usage_estimated": False,
    }
    return pd.DataFrame(
        [
            base
            | {
                "tier": "frontier",
                "model": "gemini/pro",
                "requests": 20,
                "errors": 0,
                "ttft_p50_s": 2.0,
                "latency_mean_s": 6.0,
                "output_tokens_per_s": 40.0,
                "avg_prompt_tokens": 150.0,
                "avg_completion_tokens": 400.0,
                "billing": "paid",
                "cost_per_1k_billed_usd": 5.1,
                "cost_per_1k_list_usd": 5.1,
            },
            base
            | {
                "tier": "mid",
                "model": "gemini/flash",
                "requests": 20,
                "errors": 2,
                "ttft_p50_s": 0.5,
                "latency_mean_s": 2.0,
                "output_tokens_per_s": 150.0,
                "avg_prompt_tokens": 150.0,
                "avg_completion_tokens": 200.0,
                "billing": "free",
                "cost_per_1k_billed_usd": 0.0,
                "cost_per_1k_list_usd": 0.86,
            },
        ]
    )


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "gemini/pro",
                "query_id": "a",
                "category": "summarization",
                "latency_s": 6.0,
                "ttft_s": 2.0,
                "error": None,
            },
            {
                "model": "gemini/flash",
                "query_id": "b",
                "category": "summarization",
                "latency_s": None,
                "ttft_s": None,
                "error": "RateLimitError: quota",
            },
        ]
    )


def _meta(dry_run: bool) -> RunMeta:
    return RunMeta("run-1", "2026-11-01 01:00 UTC", 20, 1, ("summarization",), "2026-11-01", dry_run, False)


def test_findings_compare_models() -> None:
    text = "\n".join(key_findings(_summary()))
    assert "flash (mid) at a median 0.50s" in text and "4.0x faster" in text
    assert "$5.10 on pro vs $0.8600 on flash (6x)" in text
    assert "flash 2/20" in text


def test_dry_run_is_labelled_and_kept_separate(tmp_path: Path) -> None:
    md = build_summary_markdown(_summary(), _runs(), _meta(dry_run=True))
    assert "DRY RUN" in md
    assert write_summary(_summary(), _runs(), _meta(dry_run=True), tmp_path).name == "SUMMARY.dry-run.md"
    assert write_summary(_summary(), _runs(), _meta(dry_run=False), tmp_path).name == "SUMMARY.md"


def test_real_run_has_no_dry_run_banner_and_lists_errors() -> None:
    md = build_summary_markdown(_summary(), _runs(), _meta(dry_run=False))
    assert "DRY RUN" not in md
    assert "## Errors" in md and "RateLimitError" in md


def test_markdown_table_escapes_pipes() -> None:
    assert "a\\|b" in markdown_table(["h"], [["a|b"]], "l")
