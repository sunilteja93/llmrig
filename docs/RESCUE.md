# LLMRig Rescue: protocol preflight (experimental)

**Status:** implemented on the feature branch, not part of PyPI v0.10.0.
This is the **first working substrate** of the Rescue initiative. It is not
an end-to-end autonomous OpenCode repair engine.

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

## What still needs to be built for the full Rescue vision

**Do not present this feature as a repaired coding agent.** The next phases are:

1. Additional versioned OpenCode tasks and independent, multi-run reliability
   measurements beyond the initial parity diagnostic.
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
