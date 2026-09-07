# Contributing to LLMRig

LLMRig welcomes focused contributions to code, tests, documentation, hardware
detection, model metadata, runtime support, and reproducible benchmark behavior.

## Product and evidence invariants

LLMRig is the compatibility and performance intelligence layer between hardware,
models, artifacts, quantization, runtimes, context, and measured performance. Keep
these facts independent:

```text
unknown != false
compatible != local
local != executable
executable != measurable
measurable != measured
measured != recommended
```

A result must say whether an important fact is verified, measured, inferred,
estimated, or unknown. Benchmark evidence outranks planning heuristics. There is no
benchmark winner without sufficient evidence, and an inconclusive result or explicit
tradeoff is valid. Never infer model quality, accuracy, or reasoning from throughput.

Discovery is not installation trust. A model found through live metadata must not
become auto-installable until its runtime identifier and provenance are reviewed and
curated. Native file association is user-supplied evidence, not independent content
attestation; structural validation does not prove model identity or weight
equivalence.

Private paths may cross only the narrow validation, inventory, and runtime invocation
seams. Retained locator state belongs in `InventoryTarget` and `ExecutionTarget`.
Paths must never enter public/shareable result structures, JSON, human output, errors,
logs, benchmark passports, exception messages, `repr`, or serialized copies.

## Architecture

`llmrig.py` is the installed CLI module and deliberately small public SDK facade. It
still contains established hardware, discovery, compatibility, runtime, benchmark,
race, decision, and command behavior. Do not broadly reorganize it as part of an
unrelated contribution.

`_llmrig/` is the private implementation namespace for incremental extraction:

- `inventory.py` observes already-local artifacts without installation or execution
- `planning.py` holds orthogonal candidate states and evidence-bearing assessments
- `privacy.py` rejects local paths and identities at public-data boundaries
- `solve.py` builds schema-versioned solve results and verification state

The stable SDK centers on `llmrig.solve(...) -> SolveResult`, `SolveCandidate`, and
the deliberate `SolveInputError` / `SolveEngineError` contracts. Do not expose
orchestration internals merely because they are convenient to import.

Default solve is observational and read-only. It must not download models, install
or start runtimes, execute inference, write passports, or scan arbitrary paths.
Verification is explicit and must use only already-local candidates.

## Development setup

LLMRig supports Python 3.9+ and has no third-party runtime dependencies. From a
source checkout, run:

```bash
python3 -m compileall -q llmrig.py _llmrig
python3 -m unittest discover -s tests -v
python3 llmrig.py --version
python3 llmrig.py --help
python3 llmrig.py solve --help
python3 llmrig.py check
git diff --check
```

`python3 llmrig.py check --online` also exercises live Hugging Face discovery; it is
optional because it requires network access and upstream availability.

For packaging changes, additionally build a wheel and sdist, inspect both, install
the wheel in a clean environment, and verify `import llmrig`, `import _llmrig`, the
console entry point, SDK read-only solve, version metadata, and empty runtime
requirements. Do not commit `build/`, `dist/`, virtual environments, or `*.egg-info`.

## Pull requests

1. Create a focused branch from the current development base.
2. Make the smallest coherent change that solves the problem.
3. Add or update tests for behavior changes.
4. Update documentation for user-visible changes.
5. Run the relevant validation above on affected platforms where practical.
6. Explain evidence sources, privacy impact, and any packaging/release impact.

Model sizes, identifiers, context limits, formats, platform support, and runtime
commands need primary/upstream sources. Do not fabricate support or add a stub
adapter merely to claim another model family or runtime.

## Curated models and runtimes

LLMRig is Qwen-first today. A curated model may be pulled automatically through the
existing Ollama setup path, so additions require an exact upstream identity,
download/package size, quantization or precision, context evidence, platform/runtime
support, input modalities, provenance links, and tests where practical.

New model families and runtimes should preserve the generic workflow and current CLI
semantics. Detection of an installed executable is distinct from runtime
availability, LLMRig execution support, model executability, and measurement
support. Do not turn any one of those facts into the others.

## Benchmark and schema changes

Changes to prompts, warmups, timing, token counts, context limits, comparison rules,
or correctness tests can invalidate comparability. Methodology changes require an
explicit benchmark/race method version change plus tests and documentation. Do not
change `race-v2`, its two-sample rule, or its 5% threshold incidentally.

Public schema changes need the same care: preserve deterministic JSON and version a
schema only when its contract actually changes. Benchmark passports are reproducible
records, not signatures, independent attestations, certifications, or proof that a
claimed measurement is true.

## Community expectations

Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Before starting a broad change, open
a focused issue describing the problem, evidence, and intended scope so maintainers
can confirm direction.
