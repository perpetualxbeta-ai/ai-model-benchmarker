from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import benchmarker
from benchmarker import (
    ModelConfig,
    ProviderConfig,
    Query,
    Settings,
    load_queries,
    provider_kwargs,
    request_kwargs,
    run_benchmark,
)
from metrics import Pricing

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Settings(runs=1, max_tokens=256, timeout_s=10, max_retries=0, seed=None, dry_run=True, sequential=False)


def test_queries_file_has_20_across_4_categories() -> None:
    data = json.loads((ROOT / "queries.json").read_text())
    queries = data["queries"]
    assert len(queries) == 20
    assert len({q["id"] for q in queries}) == 20
    counts: dict[str, int] = {}
    for q in queries:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    assert counts == {
        "summarization": 5,
        "json_extraction": 5,
        "logical_reasoning": 5,
        "code_generation": 5,
    }


def test_limit_applies_per_category() -> None:
    queries = load_queries(ROOT / "queries.json", None, 2)
    assert len(queries) == 8


def test_null_params_are_not_sent() -> None:
    model = ModelConfig("mid", "gemini/x", "gemini", Pricing(0, 0), params=(("temperature", None),))
    kwargs = request_kwargs(model, {}, SETTINGS, "q")
    assert "temperature" not in kwargs


def test_missing_or_placeholder_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProviderConfig("gemini", 2, env_key="GEMINI_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert provider_kwargs(provider) is None
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    assert provider_kwargs(provider) is None
    monkeypatch.setenv("GEMINI_API_KEY", "real-key")
    assert provider_kwargs(provider) == {"api_key": "real-key"}


def test_ollama_uses_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProviderConfig("ollama", 1, env_base_url="OLLAMA_BASE_URL", default_base_url="http://localhost:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/")
    assert provider_kwargs(provider) == {"api_base": "http://gpu-box:11434"}


async def test_one_failing_request_does_not_abort_run(monkeypatch: pytest.MonkeyPatch) -> None:
    real = benchmarker.stream_completion

    async def flaky(model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if "FAIL" in messages[-1]["content"]:
            raise ValueError("provider exploded")
        return await real(model, messages, **kwargs)

    monkeypatch.setattr(benchmarker, "stream_completion", flaky)
    model = ModelConfig("mid", "gemini/gemini-3.8-flash", "gemini", Pricing(1, 1))
    queries = [Query("ok", "c", "hello", ""), Query("bad", "c", "FAIL", "")]
    providers = {"gemini": ProviderConfig("gemini", 2)}
    records, responses = await run_benchmark([(model, {"api_key": "x"})], queries, providers, SETTINGS, "run")
    by_id = {r.query_id: r for r in records}
    assert by_id["ok"].error is None and by_id["ok"].latency_s is not None
    assert by_id["bad"].error is not None and "provider exploded" in by_id["bad"].error
    assert len(responses) == 1


def test_free_only_drops_paid_models() -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    candidates = benchmarker.load_tier_candidates(raw, ["frontier", "mid", "lightweight"], free_only=True)
    assert "frontier" not in candidates
    assert [m.model for m in candidates["mid"]] == ["gemini/gemini-3.8-flash"]
    assert [m.model for m in candidates["lightweight"]] == ["ollama/llama3.2:3b"]
