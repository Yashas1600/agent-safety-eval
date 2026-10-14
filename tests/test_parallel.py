"""Tests for the parallel episode runner."""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_safety_eval.parallel import (
    EpisodeResult,
    EpisodeSpec,
    ProviderRateLimiter,
    ProviderStats,
    RunStats,
    TransientError,
    _percentile,
    compute_stats,
    is_transient,
    run_parallel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    index: int,
    task_id: str = "task-1",
    provider: str = "mock",
    seed: int = 0,
) -> EpisodeSpec:
    return EpisodeSpec(index=index, task_id=task_id, provider=provider, seed=seed)


async def _instant_episode(spec: EpisodeSpec) -> EpisodeResult:
    """Episode that completes instantly with success."""
    return EpisodeResult(
        index=spec.index,
        task_id=spec.task_id,
        provider=spec.provider,
        seed=spec.seed,
        success=True,
        tokens_used=10,
    )


async def _slow_episode(spec: EpisodeSpec) -> EpisodeResult:
    """Episode that sleeps proportional to its index (for ordering tests)."""
    # Higher index => shorter sleep so they finish in reverse order
    delay = 0.05 * (10 - spec.index) / 10
    await asyncio.sleep(delay)
    return EpisodeResult(
        index=spec.index,
        task_id=spec.task_id,
        provider=spec.provider,
        seed=spec.seed,
        success=True,
        tokens_used=5,
    )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([3.0], 50) == 3.0
        assert _percentile([3.0], 95) == 3.0

    def test_even_distribution(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == 3.0
        # p0 = min, p100 = max
        assert _percentile(values, 0) == 1.0
        assert _percentile(values, 100) == 5.0

    def test_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0]
        p50 = _percentile(values, 50)
        assert 2.0 <= p50 <= 3.0


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


class TestTransientDetection:
    def test_transient_error_class(self):
        assert is_transient(TransientError("oops"))

    def test_timeout_error(self):
        assert is_transient(asyncio.TimeoutError())

    def test_regular_value_error(self):
        assert not is_transient(ValueError("bad input"))

    def test_regular_runtime_error(self):
        assert not is_transient(RuntimeError("something broke"))


# ---------------------------------------------------------------------------
# ProviderRateLimiter
# ---------------------------------------------------------------------------


class TestProviderRateLimiter:
    async def test_default_limit(self):
        limiter = ProviderRateLimiter(default_limit=2)
        # Should be able to acquire twice
        await limiter.acquire("mock")
        await limiter.acquire("mock")
        # Third acquire should block, check with wait_for
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire("mock"), timeout=0.05)
        limiter.release("mock")

    async def test_per_provider_independence(self):
        limiter = ProviderRateLimiter(default_limit=1)
        # Acquire one for each provider
        await limiter.acquire("provider_a")
        await limiter.acquire("provider_b")
        # provider_a is full, provider_b is full
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire("provider_a"), timeout=0.05)
        # But releasing provider_b does not help provider_a
        limiter.release("provider_b")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire("provider_a"), timeout=0.05)
        limiter.release("provider_a")

    async def test_custom_limit(self):
        limiter = ProviderRateLimiter(default_limit=1)
        limiter.set_limit("fast_provider", 3)
        await limiter.acquire("fast_provider")
        await limiter.acquire("fast_provider")
        await limiter.acquire("fast_provider")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire("fast_provider"), timeout=0.05)
        limiter.release("fast_provider")


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_basic_stats(self):
        results = [
            EpisodeResult(
                index=0, task_id="t1", provider="mock", seed=0,
                success=True, duration_sec=1.0, tokens_used=10,
            ),
            EpisodeResult(
                index=1, task_id="t2", provider="mock", seed=0,
                success=True, duration_sec=2.0, tokens_used=20,
            ),
            EpisodeResult(
                index=2, task_id="t3", provider="gemini", seed=0,
                success=True, duration_sec=3.0, tokens_used=30,
            ),
        ]
        stats = compute_stats(results, total_duration=3.5)
        assert stats.total_episodes == 3
        assert stats.total_duration_sec == 3.5
        assert stats.total_tokens == 60
        assert abs(stats.episodes_per_sec - 3 / 3.5) < 0.01
        assert stats.p50_latency_sec == 2.0
        # Per-provider
        assert "mock" in stats.per_provider_stats
        assert "gemini" in stats.per_provider_stats
        assert stats.per_provider_stats["mock"].count == 2
        assert stats.per_provider_stats["gemini"].count == 1
        assert abs(stats.per_provider_stats["mock"].avg_latency_sec - 1.5) < 0.01

    def test_empty_results(self):
        stats = compute_stats([], total_duration=0.0)
        assert stats.total_episodes == 0
        assert stats.episodes_per_sec == 0.0
        assert stats.p50_latency_sec == 0.0


# ---------------------------------------------------------------------------
# run_parallel: deterministic ordering
# ---------------------------------------------------------------------------


class TestRunParallel:
    async def test_deterministic_order(self):
        """Results are ordered by index regardless of completion order."""
        specs = [_make_spec(i) for i in range(10)]
        results, stats = await run_parallel(
            specs, _slow_episode, concurrency=10,
        )
        indices = [r.index for r in results]
        assert indices == list(range(10))

    async def test_all_succeed(self):
        specs = [_make_spec(i) for i in range(5)]
        results, stats = await run_parallel(
            specs, _instant_episode, concurrency=4,
        )
        assert len(results) == 5
        assert all(r.success for r in results)
        assert stats.total_episodes == 5
        assert stats.total_tokens == 50  # 10 per episode

    async def test_concurrency_limiting(self):
        """With concurrency=2, at most 2 episodes run at once."""
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def _tracking_episode(spec: EpisodeSpec) -> EpisodeResult:
            nonlocal active, max_active
            async with lock:
                active += 1
                if active > max_active:
                    max_active = active
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1
            return EpisodeResult(
                index=spec.index, task_id=spec.task_id,
                provider=spec.provider, seed=spec.seed, success=True,
            )

        specs = [_make_spec(i) for i in range(8)]
        await run_parallel(specs, _tracking_episode, concurrency=2)
        assert max_active <= 2

    async def test_mixed_providers(self):
        """Episodes from different providers run independently."""
        specs = [
            _make_spec(0, provider="a"),
            _make_spec(1, provider="b"),
            _make_spec(2, provider="a"),
            _make_spec(3, provider="b"),
        ]
        results, stats = await run_parallel(
            specs, _instant_episode, concurrency=4,
        )
        assert len(results) == 4
        assert "a" in stats.per_provider_stats
        assert "b" in stats.per_provider_stats

    async def test_timing_recorded(self):
        specs = [_make_spec(0)]
        results, _ = await run_parallel(specs, _instant_episode, concurrency=1)
        r = results[0]
        assert r.start_time > 0
        assert r.end_time >= r.start_time
        assert r.duration_sec >= 0


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestRetries:
    async def test_transient_error_retried(self):
        call_count = 0

        async def _flaky(spec: EpisodeSpec) -> EpisodeResult:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("temporary failure")
            return EpisodeResult(
                index=spec.index, task_id=spec.task_id,
                provider=spec.provider, seed=spec.seed,
                success=True, tokens_used=5,
            )

        specs = [_make_spec(0)]
        results, _ = await run_parallel(
            specs, _flaky, concurrency=1,
            max_retries=3, backoff_base=0.01, backoff_max=0.05,
        )
        assert results[0].success
        assert results[0].retries == 2
        assert call_count == 3

    async def test_non_transient_not_retried(self):
        call_count = 0

        async def _broken(spec: EpisodeSpec) -> EpisodeResult:
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent failure")

        specs = [_make_spec(0)]
        results, _ = await run_parallel(
            specs, _broken, concurrency=1,
            max_retries=3, backoff_base=0.01,
        )
        assert not results[0].success
        assert "ValueError" in results[0].error
        assert call_count == 1  # No retries

    async def test_exhausted_retries(self):
        call_count = 0

        async def _always_fails(spec: EpisodeSpec) -> EpisodeResult:
            nonlocal call_count
            call_count += 1
            raise TransientError("keeps failing")

        specs = [_make_spec(0)]
        results, _ = await run_parallel(
            specs, _always_fails, concurrency=1,
            max_retries=2, backoff_base=0.01, backoff_max=0.02,
        )
        assert not results[0].success
        assert "TransientError" in results[0].error
        # 1 initial + 2 retries = 3 total calls
        assert call_count == 3


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


class TestRunStatsDataclass:
    def test_provider_stats_avg(self):
        ps = ProviderStats(count=4, total_latency_sec=8.0)
        assert ps.avg_latency_sec == 2.0

    def test_provider_stats_zero(self):
        ps = ProviderStats()
        assert ps.avg_latency_sec == 0.0

    def test_run_stats_defaults(self):
        rs = RunStats()
        assert rs.total_episodes == 0
        assert rs.total_tokens == 0
        assert rs.per_provider_stats == {}
