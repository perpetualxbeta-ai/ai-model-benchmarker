"""Timing, token and cost metrics for streamed LLM completions.

All timings use ``time.perf_counter()``. A request's clock starts immediately
before the streaming call is awaited and stops on the final chunk, so client
setup, retry backoff and post-processing are never counted as latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import litellm
import pandas as pd

Billing = Literal["paid", "free", "local"]

# Fewer content chunks than this means the answer arrived in a burst (e.g. a
# provider buffered it), so there's no generation window to measure speed over.
MIN_STREAM_CHUNKS = 3


@dataclass(slots=True, frozen=True)
class Pricing:
    """USD price per 1M tokens and how the model is billed for this user."""

    input_per_1m: float
    output_per_1m: float
    billing: Billing = "paid"


@dataclass(slots=True)
class RequestMetrics:
    """Measurements for a single streamed completion."""

    ttft_s: float | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_estimated: bool = False
    reasoning_tokens: int = 0
    content_chunks: int = 0

    @property
    def visible_tokens(self) -> int:
        """Output tokens the user actually sees (completion minus hidden reasoning)."""
        return max(self.completion_tokens - self.reasoning_tokens, 0)

    @property
    def generation_s(self) -> float | None:
        """Time from the first visible token to the end of the stream."""
        if self.ttft_s is None or self.content_chunks < MIN_STREAM_CHUNKS:
            return None
        window = self.latency_s - self.ttft_s
        return window if window > 0 else None

    @property
    def output_tokens_per_s(self) -> float | None:
        """Streaming speed: visible tokens over the time after the first token.

        Reasoning tokens are excluded because thinking models generate them
        before the first visible token, outside this window; counting them
        would report thousands of tokens/s.
        """
        window = self.generation_s
        if window is None or self.visible_tokens <= 0:
            return None
        return self.visible_tokens / window


@dataclass(slots=True)
class StreamResult:
    """Text returned by a streamed completion plus its metrics."""

    text: str
    metrics: RequestMetrics
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _chunk_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None) if delta is not None else None
    return content or ""


def _chunk_finish_reason(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    reason = getattr(choices[0], "finish_reason", None)
    return str(reason) if reason else None


def _reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    value = getattr(details, "reasoning_tokens", None) if details is not None else None
    return int(value or 0)


async def stream_completion(
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> StreamResult:
    """Run one streamed completion through litellm and measure it.

    TTFT is taken on the first chunk carrying visible content; role-only,
    empty and reasoning-only deltas don't count. Usage comes from the
    provider's final stream chunk; if the provider omits it, tokens are
    estimated with litellm's tokenizer and ``usage_estimated`` is set.
    """
    kwargs.setdefault("stream_options", {"include_usage": True})

    parts: list[str] = []
    content_chunks = 0
    ttft_s: float | None = None
    usage: Any = None
    finish_reason: str | None = None

    start = time.perf_counter()
    response = await litellm.acompletion(model=model, messages=messages, stream=True, **kwargs)
    async for chunk in response:
        text = _chunk_text(chunk)
        if text:
            if ttft_s is None:
                ttft_s = time.perf_counter() - start
            parts.append(text)
            content_chunks += 1
        finish_reason = _chunk_finish_reason(chunk) or finish_reason
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
    latency_s = time.perf_counter() - start

    full_text = "".join(parts)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    estimated = False
    if usage is None or (prompt_tokens == 0 and completion_tokens == 0):
        prompt_tokens = litellm.token_counter(model=model, messages=messages)
        completion_tokens = litellm.token_counter(model=model, text=full_text) if full_text else 0
        estimated = True

    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if not estimated else 0
    if total_tokens < prompt_tokens + completion_tokens:
        total_tokens = prompt_tokens + completion_tokens

    return StreamResult(
        text=full_text,
        finish_reason=finish_reason,
        metrics=RequestMetrics(
            ttft_s=ttft_s,
            latency_s=latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usage_estimated=estimated,
            reasoning_tokens=_reasoning_tokens(usage) if usage is not None else 0,
            content_chunks=content_chunks,
        ),
    )


def list_cost_usd(prompt_tokens: float, completion_tokens: float, pricing: Pricing) -> float:
    """Cost at the provider's list price, ignoring any free tier."""
    return (prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m) / 1_000_000


def billed_cost_usd(prompt_tokens: float, completion_tokens: float, pricing: Pricing) -> float:
    """Cost actually charged: $0 for free-tier and local models."""
    if pricing.billing in ("free", "local"):
        return 0.0
    return list_cost_usd(prompt_tokens, completion_tokens, pricing)


def cost_per_1k_queries(
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    pricing: Pricing,
    *,
    billed: bool = True,
) -> float:
    """Projected USD cost of 1,000 queries with the given average token counts."""
    per_query = (billed_cost_usd if billed else list_cost_usd)(avg_prompt_tokens, avg_completion_tokens, pricing)
    return per_query * 1_000


def summarize(runs: pd.DataFrame, pricing_by_model: dict[str, Pricing]) -> pd.DataFrame:
    """Aggregate per-request rows into one summary row per model.

    Expects the columns produced by ``benchmarker.RunRecord``. Failed requests
    are counted but excluded from timing and token statistics.
    """
    rows: list[dict[str, Any]] = []
    for (tier, model), group in runs.groupby(["tier", "model"], sort=False):
        ok = group[group["error"].isna()]
        pricing = pricing_by_model[str(model)]
        avg_prompt = float(ok["prompt_tokens"].mean()) if len(ok) else 0.0
        avg_completion = float(ok["completion_tokens"].mean()) if len(ok) else 0.0
        ttft = ok["ttft_s"].dropna()
        latency = ok["latency_s"]
        streamed = ok[ok["output_tokens_per_s"].notna()]
        reasoning = streamed.get("reasoning_tokens", 0)
        window = (streamed["latency_s"] - streamed["ttft_s"]).sum()
        # Pooled tokens / pooled time: a mean of per-request ratios lets one
        # short burst dominate the figure.
        tps = float((streamed["completion_tokens"] - reasoning).sum() / window) if window > 0 else None
        avg_reasoning = float(ok["reasoning_tokens"].mean()) if len(ok) and "reasoning_tokens" in ok else 0.0
        rows.append(
            {
                "tier": tier,
                "model": model,
                "provider": group["provider"].iloc[0],
                "requests": len(group),
                "errors": len(group) - len(ok),
                "ttft_mean_s": ttft.mean() if len(ttft) else None,
                "ttft_p50_s": ttft.quantile(0.5) if len(ttft) else None,
                "ttft_p95_s": ttft.quantile(0.95) if len(ttft) else None,
                "latency_mean_s": latency.mean() if len(latency) else None,
                "latency_p50_s": latency.quantile(0.5) if len(latency) else None,
                "latency_p95_s": latency.quantile(0.95) if len(latency) else None,
                "latency_stdev_s": latency.std() if len(latency) > 1 else None,
                "output_tokens_per_s": tps,
                "avg_prompt_tokens": avg_prompt,
                "avg_completion_tokens": avg_completion,
                "avg_reasoning_tokens": avg_reasoning,
                "avg_total_tokens": float(ok["total_tokens"].mean()) if len(ok) else 0.0,
                "total_tokens": int(ok["total_tokens"].sum()),
                "usage_estimated": bool(ok["usage_estimated"].any()) if len(ok) else False,
                "billing": pricing.billing,
                "cost_per_1k_billed_usd": cost_per_1k_queries(avg_prompt, avg_completion, pricing),
                "cost_per_1k_list_usd": cost_per_1k_queries(avg_prompt, avg_completion, pricing, billed=False),
            }
        )
    return pd.DataFrame(rows)
