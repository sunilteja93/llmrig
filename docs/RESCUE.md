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

When a real agent run fails or times out, Rescue also records a
**privacy-safe structural event sequence**. It counts recognized JSONL event
categories, a bounded list of tool categories, the position of the first
recognized corrected statement, and the number/type of later events.
No arbitrary event names, tool arguments, file paths, raw text or prompt
content are retained. The terminal prints a short summary for single-task
runs, and the JSON contains the same whitelisted event telemetry for every
A/B trial.

A statement matching the expected correction is **provisional answer
evidence**. It does not prove that the agent finished a turn, ran the
corrected code, or escaped a tool/continuation loop. A subsequent timeout
remains a task failure; the diagnostic merely notes whether a correction
appeared somewhere earlier in the event stream. The event stream does
not record trustworthy per-event timestamps, and the counters cannot by
themselves establish the reason for a hang.

Rescue switches from preflight-only uncertainty to the measured agent
evaluation status after a task run, even when it fails. Its terminal
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

## Experimental agent iteration limit (opt-in)

The read-only OpenCode evaluator can request an explicit **temporary**
agentic iteration cap, matching the documented OpenCode `steps` setting.
It is useful for investigating an agent that produces an expected fix but
continues through additional steps instead of exiting.

```bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx \
  --context 4096 --agent-test --agent-steps 4 \
  --agent-timeout 90 --output rescue-4k-steps4.json
```

The cap is inserted into the `rescue-readonly` agent inside the disposable
test's `opencode.json`. It is **not** an automatic early-exit or a kill
switch, does not edit your regular OpenCode settings, and is *not* applied
without the explicit `--agent-test --agent-steps` combination. Values
from 2 to 12 are accepted. The regular agent process timeout, deny-by-default
permissions and SHA-256 file-integrity checks still apply.

This changes one independent variable compared with the otherwise same
read-only 4K fixture test: an agent iteration limit. If the run exits
normally with all checks passing, that is **evidence that the configured
step cap improved this particular test run**, not proof that context length
caused the original failure or that real-world agent reliability improved.
If it still times out, do not keep escalating resource limits blindly.

**Check loaded Ollama context separately:** Ollama's documented
`GET /api/ps` endpoint exposes `context_length` for running models.
While a test is active, a second terminal can run:

```bash
curl -sS http://127.0.0.1:11434/api/ps
```

Compare the returned `context_length` for your exact model against
the context configured in OpenCode. The value describes the loaded model,
not necessarily the entire context history of any particular OpenCode
request. An absent model or mismatched value requires more investigation.

## Experimental automatic-compaction hypothesis (opt-in)

For an agent that produces the expected answer but repeatedly starts new
steps and times out at a small **declared** context, Rescue can test whether
OpenCode's automatic session compaction is involved. This changes **only**
`compaction.auto` in the disposable OpenCode fixture. It does not change the
user's global or project configuration and does not increase Ollama's context.

```bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx \
  --context 4096 --agent-test --agent-no-auto-compact \
  --agent-timeout 90 --output rescue-4k-compaction-off.json
```

Compare this with the previous 4K run using the same model, prompt and 90s
limit without `--agent-no-auto-compact`. Any improvement remains anecdotal
until repeated. A context-overflow/provider error after disabling compaction
is itself useful diagnostic evidence, **not** a successful repair; compaction
normally protects against overflowing context. OpenCode's
`compaction.auto: false` controls automatic compaction, not all agent
continuation or provider retries. Reverting is automatic because the
disposable fixture is removed after the run.

The privacy-safe event summary, file-integrity checks, process timeout
and deny-by-default agent permissions are unchanged. This is an
**experimental diagnosis**, never an automatic user configuration change.

## Offline review of repeated compaction evidence (no inference)

Once you have run the *same* single-task Rescue fixture several times with
default compaction and with explicit `--agent-no-auto-compact`, analyze
existing receipts without any new model inference, agent execution, or
configuration changes:

```bash
llmrig rescue-evidence \
  --baseline rescue-4k-trace90.json \
  --baseline rescue-4k-timeout180.json \
  --candidate rescue-4k-compaction-off.json \
  --candidate rescue-4k-compaction-off-run2.json \
  --candidate rescue-4k-compaction-off-run3.json \
  --output rescue-compaction-assessment.json
```

This reads **only the explicitly named local JSON reports** and generates a
new privacy-filtered summary without overwriting prior reports. The command
checks for identical model, declared context, task ID, fixture SHA-256, no
agent step-limit confounds, valid integrity checks and exactly one reported
configuration difference (default versus explicitly disabled automatic
compaction). Duplicate report paths and unsupported/missing receipt fields
are rejected. Up to ten unique reports per side may be compared.

The result separately counts strict passes, timeouts with provisional
correct-fix evidence, process exits and boundary violations. For exploratory
triage only, it labels results `repeated_mitigation_observed_not_causally_verified`
when **at least two baseline runs** timed out after producing correction
evidence and **at least three candidate runs** strictly passed, with no
boundary violations. Otherwise it is `inconclusive_or_mixed_evidence`.

**These numerical thresholds are simple engineering triage criteria, not
a statistical test or approval to disable automatic compaction globally.**
All reports are self-reported and not independently signed. Trial order,
execution budgets, Ollama's effective context and other possible confounds
are not reconstructed. Automatic compaction is often needed for long agent
tasks; turning it off can cause context-window exhaustion.

**Legacy receipts:** Earlier single-task reports were produced before
Rescue started recording an explicit `auto_compaction_disabled` boolean.
The offline assessor accepts those **only as baseline** receipts,
flags how many have an unrecorded compaction policy, and treats them
as *possibly default* compaction rather than independently verified.
Candidate receipts must explicitly record `auto_compaction_disabled: true`.

The report recommendation is always
`no_automatic_or_global_compaction_change`, even if the exploratory
criterion is met. To investigate causality, perform independent alternating
on/off trials with comparable budgets and multiple workloads.

## Paired compaction A/B experiment (explicit opt-in)

The next experiment after offline receipt triage is a controlled **on/off**
comparison over the three fixed read-only coding tasks. At the **same**
OpenCode-declared context and model, the tool alternates conditions in an
A/B then B/A schedule, using fresh disposable workspaces and HOME state
per session. The conditions differ solely in whether
`compaction.auto: false` is explicitly set in the local fixture; baseline
uses OpenCode's unspecified default.

```bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx \
  --context 4096 --compaction-ab --compaction-repeats 1 \
  --agent-timeout 90 --ab-budget 600 \
  --output rescue-compaction-paired.json
```

One repeat means **6 agent sessions**, and two repeats means **12
sessions**, which can take a long time. If selecting two repeats, increase
`--ab-budget` to at least 1200 seconds when using a 90-second per-session
limit. The runner checks before each pair that the remaining total budget
can accommodate *both* full session timeouts; incomplete experiments are
never reported as complete comparisons. No task is run without explicit
`--compaction-ab`.

Each outcome records task, compaction setting, pair number, test checks,
process completion, provisional correct-fix evidence, failure category,
and privacy-filtered event counts. The aggregate distinguishes actual
completed passes from explanation-only rubric misses and timeouts. A
single failing test is not silently converted to a success because the
model proposed a correct statement.

**Interpretation:** Alternating order and equal task budgets reduce simple
confounds but do not independently observe compaction events or establish
a causal root cause, actual Ollama per-request context, statistical
significance, or safe generalization to long coding tasks. Raw model
text and fixture paths are not retained. OpenCode runs with current user
OS permissions; the disposable config and tool policy are *not* an OS
sandbox. Regardless of outcome, LLMRig never applies a persistent
compaction setting or automatically recommends disabling it globally.

## Experimental A/B coding-task verification

LLMRig can run an **opt-in exploratory A/B trial** using three fixed read-only
debugging tasks with objective, deterministic graders:

1. Parity predicate accidentally detects odd instead of even.
2. Nonnegative clamp inadvertently negates its input.
3. Prefix check accidentally uses `endswith`.

The same installed Ollama model runs in isolated temporary OpenCode sessions
with two *declared* context allocations. Each task is repeated for both
configurations, alternating execution order between tasks and repeats to
reduce basic warmup and order bias.

```bash
llmrig rescue --agent opencode --model qwen3.8:27b-mlx \
  --context 4096 --compare-context 8192 \
  --ab-test --ab-repeats 2 \
  --agent-timeout 90 --ab-budget 600 \
  --output rescue-ab.json
```

**Runtime cost:** The default experiment invokes 12 OpenCode sessions
(3 tasks × 2 contexts × 2 repeats). Each session can execute inference, and
the A/B run has a 600-second total budget by default. A run in progress
uses at most its assigned individual session timeout, which can be shorter
than the total budget. No user project is opened, and no persistent model or
agent configuration is changed. The tool never runs without `--ab-test`.

The result includes every task ID, context, repeated-trial index, pass/fail,
whitelisted check booleans, elapsed seconds and any failure category.
It also separates **strict completed task passes** from **correction evidence
seen in the agent response**, **successful process exits**, **agent timeouts**,
and **explanation-only rubric misses**. A task that produced the correct
answer but timed out remains a failed *completed task*, without being
misrepresented as an incorrect answer. Correction evidence uses a narrow
text-based rubric, so it is not independent execution of a proposed fix.

The A/B comparison uses strict completed-task success for its existing
headline and reports separate observed differences in termination and
correction evidence. None of these imply statistically reliable improvement.
An incomplete run (time budget exhausted, OpenCode missing or a policy
boundary violation) is **not** presented as a valid comparison. A boundary
violation stops subsequent tasks.

**Very important limitations:** The two OpenCode settings change the context
*advertised in the agent's local provider config*. This does not independently
verify Ollama's actual effective `num_ctx` allocation, so the experiment
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
