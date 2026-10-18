# LLMRig Rescue: protocol preflight (experimental)

**Status:** experimental implementation merged to `main`, not part of the
published PyPI v0.10.0 artifact. Install from source to try it.
It includes an opt-in, real OpenCode coding-task evaluator, but is **not**
an autonomous OpenCode repair engine.

## What this version actually does

- Detect whether the `opencode` executable is on PATH, without opening its
  configuration, launching the agent, or modifying any files.
- Contact **loopback-only** Ollama to confirm service readiness and that the
  **exact model tag is installed**. No model pull or package installation.
- Inspect Ollama's runtime-reported model architecture, quantization, and
  tool-call capability declaration. Unlike LLMRig's separate curated catalog
  format, these fields are read from the installed model's `/api/show` metadata.
- When you explicitly add `--probe`, invoke a bounded synthetic Ollama
  chat tool-call canary. The declared tool is a **fake echo specification**;
  LLMRig never executes a real tool, reads private project files, or passes
  user prompts through this test.
- Optionally repeat the same synthetic test at two requested context sizes,
  record exact passes and failures, and label any observed difference as
  **insufficient to prove an agent improvement**.
- Produce a privacy-filtered JSON result. Raw model content, private
  workspaces, local file paths and API secrets are not in the result.

## Run a preflight with NO inference

Install the current feature branch in an isolated development environment;
the published `pipx install llmrig` release does not yet contain Rescue.

```bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx
```

This contacts only the local Ollama HTTP API to inspect model metadata.
You will see `ready_for_probe` or a factual blocker such as the model not
being installed. OpenCode presence is only a PATH check. It is **not**
evidence that OpenCode can actually connect to this model.

## Explicit synthetic tool-call test

```bash
llmrig rescue --agent opencode \
  --model qwen3.8:27b-mlx \
  --context 4096 \
  --probe --runs 3
```

This causes real, local model inference and may use GPU memory. It does not
change OpenCode settings or execute the tool it asks the model to call.

## Compare two context configurations

```bash
llmrig rescue --agent opencode \
  --model qwen3.8:27b-mlx \
  --context 4096 \
  --compare-context 8192 \
  --probe --runs 3 \
  --output rescue-result.json
```

When you pass `--output`, Rescue saves the whitelisted JSON data and still
prints each context's observed pass/fail count and comparison assessment.
You do not need to open the JSON file just to see whether the test passed.
The summary does not include raw model-generated text.

The baseline and candidate test use the **same installed model** and
identical synthetic tool-call prompt, changing the requested Ollama
`num_ctx` value. The report will include observed passes, simple
latency measurements and the explicit caveat that this is *not* a
statistically established improvement. It does not persist any model
configuration or auto-select a winner.

The default is read-only preflight. `--compare-context` without
`--probe` is rejected. Maximum 5 calls per context, maximum 2 contexts.
The tool response is limited to 2 MiB. Each local inference has a 60s
request timeout. `--output` writes a new local JSON file only,
never overwriting. No report is uploaded automatically.

Only loopback HTTP endpoints (`localhost`, `127.0.0.1` or equivalent
loopback IPs) with explicit ports are supported. A configured remote
`OLLAMA_HOST` is refused.

## Automated OpenCode coding task (opt-in, experimental)

Once `opencode` is on your PATH and the exact Ollama model is installed:

```bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx \
  --context 8192 --agent-test --agent-timeout 120 \
  --output rescue-agent-result.json
```

Unlike the synthetic Ollama-only `--probe`, `--agent-test` **executes the
actual OpenCode CLI** in a newly created disposable directory. It asks the
model to read a tiny file containing an incorrect parity predicate and
explain the exact correction. No existing user project is opened.

The evaluator checks a structured `opencode run --format json` event
stream for all of the following:

- The agent invoked a completed `read` tool for the fixture.
- Its answer both explained the odd-number bug and proposed the
  exact corrected return statement.
- The fixture's SHA-256 remained unchanged.
- No other than read, glob, grep or list tool was observed.
- The agent exited successfully, emitted no error event and provided
  a valid bounded event stream.

Rescue switches from preflight-only uncertainty to the measured agent
evaluation status after a task run, even when the task fails. Its terminal
header identifies whether evidence came from preflight, a synthetic Ollama
protocol canary, or an actual OpenCode coding-task evaluation.

Reports include check booleans, duration, event count, failure category
and an SHA-256 digest of the synthetic fixture. **Raw agent text, prompts,
local paths and event contents are never included in the report.**
Reports are local only and never overwrite existing files.

**Important security limitation:** OpenCode is real third-party code
running under your operating-system user. The evaluator sets a temporary
HOME, isolated XDG directories, only the local Ollama provider, and
deny-by-default tool permissions. It does **not** impose kernel-enforced
sandboxing, block all possible outbound network access, or certify
OpenCode itself. Run only if you trust your installed OpenCode executable.
Agent session state is created inside the disposable directory and
removed after the test.

`--agent-test` is mutually exclusive with `--probe` and
`--compare-context`; it is not automatically invoked by preflight.
This first evaluator diagnoses a **single small coding task**. It does
not automatically change OpenCode's configuration, run arbitrary coding
projects, or prove a repair. Subsequent phases will add versioned
tasks, explicit counterfactual configurations and comparison evidence.

## Experimental A/B coding-task verification

LLMRig can run an **opt-in exploratory A/B trial** using three fixed read-only
debugging tasks with objective, deterministic graders:

1. Parity predicate accidentally detects odd instead of even.
2. Nonnegative clamp inadvertently negates its input.
3. Prefix check accidentally uses \`endswith\`.

The same installed Ollama model runs in isolated temporary OpenCode sessions
with two *declared* context allocations. Each task is repeated for both
configurations, alternating execution order between tasks and repeats to
reduce basic warmup and order bias.

\`\`\`bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx \
  --context 4096 --compare-context 8192 \
  --ab-test --ab-repeats 2 \
  --agent-timeout 90 --ab-budget 600 \
  --output rescue-ab.json
\`\`\`

**Runtime cost:** The default experiment invokes 12 OpenCode sessions
(3 tasks × 2 contexts × 2 repeats). Each session can execute inference, and
the A/B run has a 600-second total budget by default. A run in progress
uses at most its assigned individual session timeout, which can be shorter
than the total budget. No user project is opened, and no persistent model or
agent configuration is changed. The tool never runs without \`--ab-test\`.

The result includes every task ID, context, repeated-trial index, pass/fail,
whitelisted check booleans, elapsed seconds and any failure category. It
also shows simple aggregate success proportions and the observed difference.
An incomplete run (time budget exhausted, OpenCode missing or a policy
boundary violation) is **not** presented as a valid comparison. A boundary
violation stops subsequent tasks.

**Very important limitations:** The two OpenCode settings change the context
*advertised in the agent's local provider config*. This does not independently
verify Ollama's actual effective \`num_ctx\` allocation, so the experiment
does not establish true context-size effects. Prompts are intentionally
short and do not stress long-context capacity. The fixed tiny tasks are
not representative coding benchmarks, and repeated results from one
machine do not establish statistical significance. LLMRig **never**
automatically selects a winner, changes the real agent configuration,
or claims to have repaired an agent based on these small comparisons.

Temporary homes and restrictive tool policy reduce exposure but do not
constitute an operating-system sandbox. OpenCode is still a real
third-party executable running with the invoking user's privileges.
Results are stored locally and raw agent output is never included
in reports.

## What still needs to be built for the full Rescue vision

**Do not present this feature as a repaired coding agent.** The next phases are:

1. Representative larger agent tasks and independent reliability studies,
   beyond the current three-task exploratory A/B harness.
2. Versioned code-edit tasks with independently checked outputs and a
   true baseline-versus-candidate task success comparison.
3. Supported reversible configuration changes with approval, restore
   receipts, and resource guards. Never bypass authentication, model
   security policies or tool-use permission controls.
4. Cross-runtime probes and repeatability on additional machines.
5. Disclose evidence gaps and confidently say **no supported repair found**
   when the evidence does not justify a change.

The motivation is **diagnose → reproduce → test a hypothesis → verify**,
not "make all agents work" or "automatic agent repair" without tests.
