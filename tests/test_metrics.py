from __future__ import annotations

import pandas as pd
import pytest

from metrics import (
    Pricing,
    RequestMetrics,
    billed_cost_usd,
    cost_per_1k_queries,
    list_cost_usd,
    stream_completion,
    summarize,
)

PAID = Pricing(input_per_1m=2.0, output_per_1m=12.0, billing="paid")
FREE = Pricing(input_per_1m=0.75, output_per_1m=3.75, billing="free")
LOCAL = Pricing(input_per_1m=0.0, output_per_1m=0.0, billing="local")


def test_list_cost() -> None:
    # 1,000 in @ $2/M + 500 out @ $12/M = 0.002 + 0.006
    assert list_cost_usd(1_000, 500, PAID) == pytest.approx(0.008)


def test_cost_per_1k_queries_paid() -> None:
    assert cost_per_1k_queries(1_000, 500, PAID) == pytest.approx(8.0)


def test_free_tier_billed_is_zero_but_list_price_reported() -> None:
    assert billed_cost_usd(1_000, 500, FREE) == 0.0
    assert cost_per_1k_queries(1_000, 500, FREE, billed=False) == pytest.approx(2.625)


def test_local_is_zero_both_ways() -> None:
    assert cost_per_1k_queries(1_000, 500, LOCAL) == 0.0
    assert cost_per_1k_queries(1_000, 500, LOCAL, billed=False) == 0.0


def test_throughput_excludes_ttft() -> None:
    m = RequestMetrics(
        ttft_s=0.5, latency_s=2.5, prompt_tokens=10, completion_tokens=100, total_tokens=110, content_chunks=20
    )
    assert m.output_tokens_per_s == pytest.approx(50.0)


def test_throughput_none_without_generation_window() -> None:
    m = RequestMetrics(ttft_s=1.0, latency_s=1.0, prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert m.output_tokens_per_s is None


async def test_stream_completion_measures_ttft_and_usage() -> None:
    result = await stream_completion(
        "gemini/gemini-3.8-flash",
        [{"role": "user", "content": "hi"}],
        mock_response="Hello from the mock response.",
        api_key="test",
    )
    m = result.metrics
    assert result.text == "Hello from the mock response."
    assert m.ttft_s is not None and 0 <= m.ttft_s <= m.latency_s
    assert m.prompt_tokens > 0 and m.completion_tokens > 0
    assert m.total_tokens == m.prompt_tokens + m.completion_tokens
    assert m.usage_estimated is False


def test_summarize_excludes_errors_from_stats() -> None:
    base = {
        "tier": "mid",
        "model": "m",
        "provider": "p",
        "ttft_s": 0.2,
        "latency_s": 1.0,
        "output_tokens_per_s": 50.0,
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "usage_estimated": False,
        "error": None,
    }
    failed = base | {
        "ttft_s": None,
        "latency_s": None,
        "output_tokens_per_s": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "error": "boom",
    }
    df = pd.DataFrame([base, base | {"latency_s": 3.0}, failed])
    row = summarize(df, {"m": PAID}).iloc[0]
    assert row["requests"] == 3 and row["errors"] == 1
    assert row["latency_mean_s"] == pytest.approx(2.0)
    assert row["avg_prompt_tokens"] == pytest.approx(100)
    assert row["cost_per_1k_billed_usd"] == pytest.approx((100 * 2 + 40 * 12) / 1_000)


def test_throughput_ignores_reasoning_and_bursts() -> None:
    thinking = RequestMetrics(
        ttft_s=40.0,
        latency_s=40.5,
        prompt_tokens=10,
        completion_tokens=700,
        total_tokens=710,
        reasoning_tokens=600,
        content_chunks=10,
    )
    assert thinking.output_tokens_per_s == pytest.approx(200.0)
    burst = RequestMetrics(
        ttft_s=2.7, latency_s=2.7037, prompt_tokens=10, completion_tokens=76, total_tokens=86, content_chunks=1
    )
    assert burst.output_tokens_per_s is None
