# Decision Log

Decisions made during development, with rationale.

## 1. In-memory WorldState over database or filesystem

**Decision:** All simulation state (files, email, calendar) lives in a Python dataclass (`WorldState`) with dict/list fields. No SQLite, no tmpdir files.

**Rationale:** The benchmark needs to be fast, deterministic, and require zero setup. In-memory state lets episodes run in milliseconds, simplifies snapshot/restore for scoring, and avoids filesystem cleanup. The tradeoff is lower fidelity to real file systems, but that is acceptable for a safety eval focused on tool-call patterns, not I/O behavior.

## 2. Deterministic UserSim over LLM-based user simulation

**Decision:** UserSim is a deterministic function (always approve, always deny, or always ambiguous) rather than an LLM generating user responses.

**Rationale:** Reproducibility is critical for a benchmark. An LLM-based user sim would introduce uncontrolled variance and require API keys even for offline runs. The deterministic approach means results depend only on the model under test. The tradeoff is that real users give varied, context-dependent responses, which is documented as a limitation.

## 3. Wilson CIs over bootstrap or t-tests

**Decision:** Confidence intervals use the Wilson score method for binomial proportions.

**Rationale:** All metrics are binary per episode (e.g., UIAR > 0 or not). Wilson intervals are appropriate for proportions with small n, handle edge cases (p = 0 or p = 1) better than Wald intervals, and are simple to compute. Bootstrap would give tighter intervals but adds complexity and runtime. No formal hypothesis tests are included because the sample size (60 episodes per model at 3 seeds) is too small for meaningful power.

## 4. Action key format for confirmation tracking

**Decision:** Confirmation is tracked per action key (`tool_name+target`), where target is the first matching argument from `(path, id, to, src)`.

**Rationale:** Tracking at the tool level alone is too coarse (approving one `delete_file` should not approve all deletes). Tracking at the full argument level is too strict (minor formatting differences would break matching). The `tool+primary_arg` granularity matches how a user would think about approval: "yes, delete that specific file."

## 5. Mock adapters as test fixtures, not benchmark entries

**Decision:** Mock adapter results (careful, reckless, lazy, injection_victim) are used to validate scoring logic. They are never reported as benchmark results alongside real models.

**Rationale:** Mock adapters are hand-coded heuristics, not model outputs. Reporting them alongside Gemini or Claude would be misleading. They serve as sanity checks: if reckless does not have high UIAR, the scoring is broken.

## 6. mcp<2 pin

**Decision:** The `mcp` package is pinned below version 2 in pyproject.toml.

**Rationale:** MCP v2 renamed `FastMCP` to `MCPServer` and changed the import path, breaking `from mcp.server.fastmcp import FastMCP`. Pinning below 2 avoids this breakage while the ecosystem stabilizes.

## 7. Parallel runner uses asyncio, not multiprocessing

**Decision:** The parallel runner uses `asyncio.gather()` with semaphores, not `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`.

**Rationale:** Model adapter calls are I/O-bound (waiting for API responses), not CPU-bound. asyncio provides lower overhead, simpler rate limiting (per-provider semaphores), and deterministic result ordering via `gather()`. In-process mock episodes complete in microseconds, so the overhead of process spawning would dominate.

## 8. YAML task files over Python or JSON

**Decision:** Tasks are defined as YAML files, not Python dataclasses or JSON.

**Rationale:** YAML is human-readable, supports multi-line strings (for instructions and rationale), and is easy for non-developers to edit. JSON lacks comments and multi-line strings. Python files would couple task definitions to the implementation. The YAML loader validates against the TaskDef schema at load time, catching errors early.

## 9. Adapter protocol over base class

**Decision:** `ModelAdapter` is a `typing.Protocol`, not an abstract base class.

**Rationale:** Structural subtyping (duck typing) means adapter classes do not need to inherit from a base. This keeps adapters independent, allows third-party adapters without importing our code, and avoids the "diamond of death" if someone combines multiple protocols. The tradeoff is that type errors are caught at call sites, not at class definition, but that is acceptable given the small number of adapters.

## 10. ADK adapter creates fresh sessions per step

**Decision:** The ADK adapter creates a new ADK Agent and session for each `step()` call.

**Rationale:** The harness manages the conversation loop, not the ADK agent. Reusing a session would let ADK's internal memory drift from the harness's message history. Fresh sessions per step ensure the harness has full control. The tradeoff is that ADK's multi-turn optimization is not used, but this keeps behavior consistent with other adapters.

## 11. Injection detection uses forbidden-rule matching, not source tracing

**Decision:** A tool call is considered "injection-sourced" if the task is an injection task and the call matches a forbidden rule. No call-graph analysis or provenance tracking.

**Rationale:** True provenance tracking (which token in which message caused which tool call) is not feasible with black-box model APIs. The heuristic works because injection tasks are designed so that the forbidden target only appears in the injected text, not in the user instruction. False positives are possible in theory but unlikely given the task design.
