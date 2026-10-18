# LLMRig v0.11.0 | Rescue: Diagnose Local Coding-Agent Failures

LLMRig v0.11.0 adds the **experimental Rescue workflow** for local Ollama
models used through OpenCode. It distinguishes a model producing an expected
correction from the agent **actually completing** its turn.

## What's new

- `llmrig rescue`: read-only preflight for a named Ollama model and OpenCode.
  No model inference or OpenCode execution without the explicit test flags.
- `--agent-test`: opt-in, disposable read-only debugging fixture with
  exact file-integrity and tool-permission checks.
- `--probe`: optional synthetic protocol canary, explicitly not a full
  coding-agent benchmark.
- `--ab-test`: fixed three-task comparison of *declared* OpenCode contexts.
- `--compaction-ab`: alternating baseline/disabled automatic compaction
  experiment across three fixed tasks, same declared context and timeouts.
- `--agent-no-auto-compact` and `--agent-steps`: isolated, opt-in
  hypotheses for a single disposable test, never persistent user changes.
- `llmrig rescue-evidence`: offline, privacy-filtered receipt comparison
  with explicit unknowns, including support for older baseline receipts.
- Privacy-preserving event sequencing that differentiates correction
  evidence, agent exits, timeouts and explanation-rubric misses.

## Measured experimental case

**Environment:** One user-reported Apple M4 Max (48 GiB), OpenCode
1.19.3, Ollama 0.33.3 and `qwen3.8:27b-mlx`.

At an **OpenCode-declared 4096-token context**, a matched,
alternating-order three-task comparison recorded:

| Compaction policy | Strict completed tasks | Timeouts | Correct-fix evidence |
| --- | ---: | ---: | ---: |
| OpenCode default | 0/3 | 3 | 3/3 |
| Explicitly disabled in disposable fixture | 3/3 | 0 | 3/3 |

Earlier independent repetitions also had 3/3 successful runs with
compaction explicitly disabled (28.43 s, 22.36 s and 10.21 s).
All results were self-reported by one local machine, with small synthetic
tasks. The matched experiment is **exploratory**, not statistical proof
of a broad improvement or an independently diagnosed root cause.
No direct OpenCode compaction events or effective Ollama per-request
context were measured.

Rescue labels the narrow failure pattern **suspected
compaction-associated post-answer continuation**, but its default
recommendation remains **no automatic or global compaction change**.
Disabling compaction in long coding sessions may cause context exhaustion.

## Install and try

```bash
pipx upgrade llmrig
llmrig --version
llmrig rescue --agent opencode --model YOUR_OLLAMA_MODEL

# Only if you trust your installed OpenCode and want the explicit real-agent test
llmrig rescue --agent opencode --model YOUR_OLLAMA_MODEL \
  --context 4096 --agent-test --agent-timeout 90
```

Details: [Rescue guide](https://github.com/sunilteja93/llmrig/blob/main/docs/RESCUE.md).

## Security and limits

- OpenCode is third-party code running with your OS user privileges.
  Temporary HOME, restricted tools and isolated workspaces are **not**
  an OS-level sandbox or network firewall.
- No user project is opened, no permanent configuration is updated, no
  raw agent messages are included in Rescue reports, and reports stay local.
- Do not interpret text-matched expected fixes as verified code execution,
  or infer general agent reliability from the tiny included tasks.
- Version v0.11.0 keeps the same Python 3.9+ baseline and dependency contract.

Full changelog:
[compare v0.10.0...v0.11.0](https://github.com/sunilteja93/llmrig/compare/v0.10.0...v0.11.0).
