"""Parallel episode runner with concurrency control, rate limiting, and stats.

Runs episode callables concurrently using asyncio, with per-provider
rate limiting, retries with exponential backoff, and deterministic
result ordering (by submission index, not completion order).
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSpec:
    """Specification for a single episode to run.

    Attributes:
        index: Position in the submission list (used for deterministic ordering).
        task_id: Identifier for the task.
        provider: Provider string (e.g. "gemini", "anthropic", "mock").
        seed: Random seed for this episode.
        config: Arbitrary adapter/task config passed through to the runner.
    """

    index: int
    task_id: str
    provider: str
    seed: int = 0
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result from running a single episode.

    Attributes:
        index: Original submission index (for deterministic ordering).
        task_id: Task identifier.
        provider: Provider string.
        seed: Seed used.
        success: Whether the episode completed without error.
        error: Error message if the episode failed after all retries.
        data: Arbitrary result data returned by the episode callable.
        start_time: Epoch timestamp when the episode started.
        end_time: Epoch timestamp when the episode finished.
        duration_sec: Wall-clock duration in seconds.
        tokens_used: Number of tokens consumed (0 if not tracked).
        retries: Number of retry attempts before success or final failure.
    """

    index: int
    task_id: str
    provider: str
    seed: int
    success: bool
    error: str | None = None
    data: Any = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_sec: float = 0.0
    tokens_used: int = 0
    retries: int = 0


@dataclass
class ProviderStats:
    """Aggregate stats for a single provider."""

    count: int = 0
    total_latency_sec: float = 0.0

    @property
    def avg_latency_sec(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_latency_sec / self.count


@dataclass
class RunStats:
    """Aggregate stats for a full parallel run."""

    total_episodes: int = 0
    total_duration_sec: float = 0.0
    episodes_per_sec: float = 0.0
    p50_latency_sec: float = 0.0
    p95_latency_sec: float = 0.0
    total_tokens: int = 0
    per_provider_stats: dict[str, ProviderStats] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


class TransientError(Exception):
    """Raised to signal a transient/retryable failure."""


def is_transient(exc: BaseException) -> bool:
    """Return True if the exception looks transient and worth retrying."""
    if isinstance(exc, TransientError):
        return True
    # Common HTTP-level transient errors from API clients
    name = type(exc).__name__
    if name in ("RateLimitError", "APIConnectionError", "InternalServerError"):
        return True
    # asyncio.TimeoutError is transient
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    # OSError with errno for connection reset, etc.
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (
        104,  # ECONNRESET
        110,  # ETIMEDOUT
        111,  # ECONNREFUSED
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Rate limiter (per-provider semaphore)
# ---------------------------------------------------------------------------


class ProviderRateLimiter:
    """Per-provider concurrency limiter using asyncio.Semaphore.

    Each provider gets its own semaphore so that one slow provider
    does not starve others.
    """

    def __init__(self, default_limit: int = 4) -> None:
        self._default_limit = default_limit
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._limits: dict[str, int] = {}

    def set_limit(self, provider: str, limit: int) -> None:
        """Set the concurrency limit for a specific provider."""
        self._limits[provider] = limit
        # Recreate semaphore if already exists
        self._semaphores.pop(provider, None)

    def _get_semaphore(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._semaphores:
            limit = self._limits.get(provider, self._default_limit)
            self._semaphores[provider] = asyncio.Semaphore(limit)
        return self._semaphores[provider]

    async def acquire(self, provider: str) -> None:
        sem = self._get_semaphore(provider)
        await sem.acquire()

    def release(self, provider: str) -> None:
        sem = self._get_semaphore(provider)
        sem.release()


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile value from a sorted list using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # Use the "exclusive" method: rank = pct/100 * (n + 1) - 1 (0-indexed)
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


# Type alias for the episode callable.
# It receives an EpisodeSpec and returns an EpisodeResult (or raises).
EpisodeCallable = Callable[[EpisodeSpec], Awaitable[EpisodeResult]]


async def _run_one(
    spec: EpisodeSpec,
    episode_fn: EpisodeCallable,
    rate_limiter: ProviderRateLimiter,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
) -> EpisodeResult:
    """Run a single episode with rate limiting and retry logic."""
    retries = 0
    last_error: BaseException | None = None

    while True:
        await rate_limiter.acquire(spec.provider)
        start = time.monotonic()
        start_epoch = time.time()
        try:
            result = await episode_fn(spec)
            end = time.monotonic()
            end_epoch = time.time()
            result.start_time = start_epoch
            result.end_time = end_epoch
            result.duration_sec = end - start
            result.retries = retries
            return result
        except Exception as exc:
            end = time.monotonic()
            end_epoch = time.time()
            last_error = exc
            if is_transient(exc) and retries < max_retries:
                retries += 1
                delay = min(backoff_base * (2 ** (retries - 1)), backoff_max)
                await asyncio.sleep(delay)
                continue
            # Non-transient or exhausted retries
            return EpisodeResult(
                index=spec.index,
                task_id=spec.task_id,
                provider=spec.provider,
                seed=spec.seed,
                success=False,
                error=f"{type(last_error).__name__}: {last_error}",
                start_time=start_epoch,
                end_time=end_epoch,
                duration_sec=end - start,
                retries=retries,
            )
        finally:
            rate_limiter.release(spec.provider)


def compute_stats(results: list[EpisodeResult], total_duration: float) -> RunStats:
    """Compute aggregate run statistics from a list of episode results."""
    durations = [r.duration_sec for r in results]
    total_episodes = len(results)
    total_tokens = sum(r.tokens_used for r in results)

    per_provider: dict[str, ProviderStats] = {}
    for r in results:
        if r.provider not in per_provider:
            per_provider[r.provider] = ProviderStats()
        ps = per_provider[r.provider]
        ps.count += 1
        ps.total_latency_sec += r.duration_sec

    eps_per_sec = total_episodes / total_duration if total_duration > 0 else 0.0

    return RunStats(
        total_episodes=total_episodes,
        total_duration_sec=total_duration,
        episodes_per_sec=eps_per_sec,
        p50_latency_sec=_percentile(durations, 50),
        p95_latency_sec=_percentile(durations, 95),
        total_tokens=total_tokens,
        per_provider_stats=per_provider,
    )


async def run_parallel(
    specs: list[EpisodeSpec],
    episode_fn: EpisodeCallable,
    concurrency: int = 4,
    rate_limiter: ProviderRateLimiter | None = None,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
) -> tuple[list[EpisodeResult], RunStats]:
    """Run episodes in parallel with concurrency control.

    Args:
        specs: List of episode specifications to run.
        episode_fn: Async callable that runs a single episode.
        concurrency: Maximum number of episodes running at once.
        rate_limiter: Optional per-provider rate limiter. If None, a default
            limiter is created with ``concurrency`` as the per-provider limit.
        max_retries: Maximum retry attempts for transient errors.
        backoff_base: Base delay (seconds) for exponential backoff.
        backoff_max: Maximum backoff delay (seconds).

    Returns:
        A tuple of (results, stats) where results are ordered by spec index
        (deterministic ordering regardless of completion order).
    """
    if rate_limiter is None:
        rate_limiter = ProviderRateLimiter(default_limit=concurrency)

    global_sem = asyncio.Semaphore(concurrency)

    async def _wrapped(spec: EpisodeSpec) -> EpisodeResult:
        async with global_sem:
            return await _run_one(
                spec,
                episode_fn,
                rate_limiter,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
            )

    wall_start = time.monotonic()

    # Launch all tasks and gather results. asyncio.gather preserves
    # the order of the input awaitables, so results come back in
    # submission order regardless of completion order.
    tasks = [asyncio.create_task(_wrapped(spec)) for spec in specs]
    results: list[EpisodeResult] = list(await asyncio.gather(*tasks))

    wall_end = time.monotonic()
    total_duration = wall_end - wall_start

    # Sort by index as a safety net (gather already preserves order)
    results.sort(key=lambda r: r.index)

    stats = compute_stats(results, total_duration)
    return results, stats
