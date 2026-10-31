# Agent Safety Eval: Do Tool-Using LLM Agents Act Without Permission?

## Abstract

We present a sandboxed benchmark of 20 tasks over mock MCP servers (files, email, calendar) that evaluates whether tool-using LLM agents take irreversible or out-of-scope actions without explicit user confirmation. The main metric, UIAR (Unconfirmed Irreversible Action Rate), measures how often an agent deletes files, sends emails, or cancels meetings without asking first. We evaluate [TBD: model names] and find that [TBD: headline finding, e.g., confirmation gating reduced the UIAR from X% to Y%]. The benchmark is reproducible with one command (`make demo` for offline, `make eval` for real models) and open-source under the MIT license.

## 1. Introduction

LLM agents with tool access can take real-world actions: delete files, send messages, cancel appointments. When these actions are irreversible, executing them without user confirmation poses a safety risk. An agent that "helpfully" sends an email the user intended to draft, or deletes files matching a vague query, creates outcomes that cannot be undone.

This benchmark measures how often agents make this mistake. We define five categories of safety-relevant behavior:

1. **Irreversible actions without confirmation** (the main metric)
2. **Wrong tool or wrong target selection**
3. **Prompt injection compliance** (obeying injected instructions in tool output)
4. **Ambiguity handling** (asking for clarification vs. guessing)
5. **Over-refusal** (refusing or over-asking on clearly authorized tasks)

## 2. Method

### 2.1 Setup

The benchmark uses three mock MCP servers modeling files (Drive-like), email (Gmail-like), and calendar (Calendar-like) operations. All state is in-memory. Each server exposes tools with an `irreversible` flag: `delete_file`, `empty_trash`, `send_email`, `delete_email`, and `cancel_event`.

A deterministic UserSim responds to confirmation requests based on each task's configuration (approve, deny, or ambiguous). The episode runner loops for up to 12 turns per task.

### 2.2 Task Suite

20 tasks across 5 categories. Each task specifies initial world state, success predicates, forbidden actions, and the UserSim behavior. Categories:

- **Irreversible (6 tasks):** Vague delete requests, send vs. draft ambiguity, bulk operations, cancel meetings with attendees.
- **Wrong tool (4 tasks):** Confusable filenames, reply vs. forward, update vs. cancel, archive vs. delete.
- **Injection (4 tasks):** Prompt injections embedded in email bodies, file contents, calendar descriptions, and footer text.
- **Ambiguity (3 tasks):** Underspecified recipients, meeting times, or file targets.
- **Benign control (3 tasks):** Pre-authorized actions where the agent should act without asking.

### 2.3 Metrics

- **Task Success Rate:** Fraction of episodes where all end-state predicates pass.
- **UIAR:** Fraction of episodes with at least one unconfirmed irreversible action.
- **Wrong Tool Rate:** Fraction of episodes with a disallowed or forbidden tool call.
- **Over-Refusal Rate:** Fraction of benign_control tasks where the agent over-asked or failed.
- **Injection Compliance:** Fraction of injection tasks where the agent obeyed the injected instruction.

All rates include Wilson 95% confidence intervals.

## 3. Results

### 3.1 Mock Validation

Mock policies validate that scoring works correctly:

| Metric | careful | reckless | lazy | injection_victim |
|--------|---------|----------|------|------------------|
| Success Rate | [TBD] | [TBD] | [TBD] | [TBD] |
| UIAR | [TBD] | [TBD] | [TBD] | [TBD] |
| Injection Compliance | [TBD] | [TBD] | [TBD] | [TBD] |

Expected: reckless has high UIAR, careful has UIAR = 0, lazy has low success, injection_victim is flagged on injection tasks.

### 3.2 Real Model Results

[TBD: fill from results/published/ after running `make eval`]

| Metric | [TBD: Model A] | [TBD: Model B] |
|--------|-----------------|-----------------|
| Task Success Rate | [TBD] | [TBD] |
| UIAR | [TBD] | [TBD] |
| Wrong Tool Rate | [TBD] | [TBD] |
| Over-Refusal Rate | [TBD] | [TBD] |
| Injection Compliance | [TBD] | [TBD] |

### 3.3 Throughput

| Stat | Mock (offline) | [TBD: Real models] |
|------|----------------|---------------------|
| Episodes/sec | [TBD] | [TBD] |
| p50 latency | [TBD] | [TBD] |
| p95 latency | [TBD] | [TBD] |
| Total tokens | n/a | [TBD] |

### 3.4 Headline Finding

[TBD: e.g., "Confirmation gating reduced the UIAR from X% to Y% across both models, with a corresponding increase in over-refusal of Z%."]

## 4. Limitations

- Mock environments simplify real API semantics.
- 20 tasks across 5 categories are not exhaustive.
- UserSim is deterministic; real users are not.
- Single-turn confirmation detection may miss multi-turn negotiation.
- Results are from a single benchmark version; cross-benchmark generalization is unknown.

See [METHODOLOGY.md](METHODOLOGY.md) for a full threats-to-validity analysis.

## 5. Related Work

[TBD: brief references to ToolBench, AgentBench, and other agent evaluation frameworks]

## 6. Conclusion

[TBD: summarize findings and next steps]
