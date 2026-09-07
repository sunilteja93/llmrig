<h1 align="center">LLMRig</h1>

<p align="center"><strong>Know what your rig can run.</strong></p>

<p align="center">The open compatibility and performance intelligence layer for local AI.</p>

<p align="center">
  <a href="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <a href="https://pypi.org/project/llmrig/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/llmrig"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sunilteja93/llmrig/main/assets/llmrig-terminal.svg" alt="LLMRig detect, resolve, assess, solve, and verify flow" width="100%" />
</p>

LLMRig determines viable ways to run AI models on real hardware. It keeps model,
artifact, quantization, runtime, context, and measured performance separate so
missing evidence does not become a confident recommendation.

Install the isolated CLI and run the safest useful first command:

```bash
pipx install llmrig
llmrig solve qwen3:0.6b
```

Default `solve` is read-only: it does not download a model, install or start a
runtime, execute inference, or write benchmark output. Measurement is an explicit
step with `--verify`, and only already-local candidates are eligible.

```text
hardware
  → model and artifact resolution
  → runtime compatibility and local availability
  → read-only solve
  → optional measured verification
  → reproducible evidence
```

LLMRig is Qwen-first today. Curated setup uses Ollama. Local race measurement also
supports llama.cpp and MLX-LM when compatible artifacts and runtimes are already
present; LLMRig does not install those native runtimes or download their artifacts.

## Install

[`pipx`](https://pipx.pypa.io/) is recommended for the CLI:

```bash
pipx install llmrig
pipx upgrade llmrig
llmrig --version
```

If `pipx` is unavailable, use a virtual environment. This also avoids PEP 668
errors from system-managed Python installations.

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install llmrig
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install llmrig
```

LLMRig supports Python 3.9+ on macOS, Linux, and Windows and has no third-party
Python runtime dependencies. It can also run directly from a source checkout with
`python3 llmrig.py`.

## Solve first

Analyze a curated model or exact `owner/repository` Hugging Face identifier:

```bash
llmrig solve MODEL
llmrig solve MODEL --json
llmrig solve MODEL --context 32768
llmrig solve MODEL --local-artifact llama.cpp=/path/to/model.gguf
llmrig solve MODEL --local-artifact mlx-lm=/path/to/model-directory
```

Hugging Face resolution reads metadata and file listings but does not download
weights. Native artifacts are inspected only at paths explicitly supplied with
`--local-artifact`; LLMRig does not scan arbitrary filesystem locations.

Solve reports these evidence dimensions independently:

```text
compatible != local
local != executable
executable != measurable
measurable != measured
measured != recommended
```

Unknown is not false. An observed Ollama name is local evidence, not independent
content attestation. A native model association is supplied by the user: LLMRig
checks its local structure but does not attest its identity, weights, quality, or
quantization. Private native paths stay out of public solve results.

Solve does not calculate a universal score, infer quality from throughput, or
promise a winner. It recommends a planning candidate only when the available
evidence supports exactly one runnable configuration with no comparable unresolved
alternative. An inconclusive result is valid.

### Verify explicitly

```bash
llmrig solve MODEL \
  --local-artifact llama.cpp=/path/to/model.gguf \
  --verify
```

`solve --verify` executes only compatible, already-local, executable, measurable
candidates. It requires at least two comparable configurations and reuses the
unchanged `race-v2` workload, two-sample rule, 5% comparison threshold, and balanced
Pareto decision. It does not pull models, install or start runtimes, scan for
artifacts, or write passports. Fewer than two candidates makes verification
unavailable without executing inference; a competitor failure invalidates the
comparison; multiple Pareto members remain inconclusive.

Solve exit codes are:

- `0`: analysis completed, including a completed but inconclusive verification
- `1`: an operational solve or attempted verification failure
- `2`: invalid/unresolvable input or unavailable requested verification

## Minimal Python SDK

```python
import llmrig

result = llmrig.solve("qwen3:0.6b")
print(result.plan.recommendation_status)
```

The stable entry point is:

```python
llmrig.solve(
    model,
    *,
    context=None,
    local_artifacts=(),
    verify=False,
) -> llmrig.SolveResult
```

It prints nothing and never exits the process. `local_artifacts` accepts
`RUNTIME=PATH` strings. Invalid or unresolvable input raises `SolveInputError`;
operational failures raise `SolveEngineError`. `SolveResult` and `SolveCandidate`
are the deliberate public result contracts. `_llmrig` is private implementation.

## Measurement commands

The measurement commands share local execution machinery but answer different
questions:

| Command | Purpose | Result rule | Writes passports |
|---|---|---|---|
| `solve --verify` | Verify a solve candidate set | Balanced Pareto decision, or inconclusive | No |
| `race` | Measure at least two local configurations | Metric-specific generation, prompt, and latency results | Optional |
| `choose` | Explain one measured objective | `generation`, `prompt`, `latency`, or balanced | No |
| `optimize` | Expose measured tradeoffs | Unranked noise-aware Pareto frontier | No |
| `bench` | Run the full Ollama benchmark | Throughput, residency data when available, and smoke tests | Optional |

Race two already-local native configurations, or combine one with an installed
equivalent Ollama build:

```bash
llmrig race MODEL \
  --local-artifact llama.cpp=/path/to/model.gguf \
  --local-artifact mlx-lm=/path/to/model-directory
```

`race` never installs or downloads. At most one explicit artifact per native
runtime is accepted. A GGUF target must be a non-empty `.gguf` file. An MLX-LM
target must have an immediate non-empty `config.json` and at least one immediate
non-empty `model*.safetensors` file. These structural checks do not prove the
runtime can load the artifact or that differently packaged artifacts have identical
weights or quality.

Race exit `0` means at least two competitors were measured successfully, `1` means
execution failed and invalidated the comparison, and `2` means the race is
unavailable or unresolved. Results within 5% are inconclusive. There is no
composite result or model-quality claim.

Choose an explicit objective from the same measured race path:

```bash
llmrig choose MODEL --objective generation
llmrig choose MODEL --objective prompt --json
llmrig choose MODEL --objective latency
llmrig choose MODEL --objective balanced
```

The default balanced objective recommends only a unique measured-performance
Pareto leader. `choose` exit `0` means a recommendation exists, `1` means race
execution failed, and `2` means the decision is unavailable or inconclusive.

Inspect the unranked frontier directly:

```bash
llmrig optimize MODEL
llmrig optimize MODEL --json
```

The active dimensions are generation throughput, prompt-evaluation throughput,
and normalized inference latency. A dimension is omitted globally if any successful
competitor lacks two positive finite samples. Missing data is never converted to
zero. The frontier measures performance only; it does not infer quality, accuracy,
or reasoning.

Run the Ollama-specific full benchmark:

```bash
llmrig bench --model qwen3.8:27b-mlx --context 32768 --runs 2
llmrig bench --all-installed --context 32768 --runs 2
llmrig bench --model qwen3.8:27b-mlx --passport benchmark.passport.json
```

`bench` unloads resident Ollama models before measurement and unloads its target
afterward. It records throughput, Ollama-reported context, accelerator residency
when available, memory snapshots, and three lightweight correctness smoke tests.
Reports default to the ignored `benchmarks/` directory. They are local
configuration checks, not academic quality benchmarks, and memory readings are
snapshots rather than peak measurements.

## Benchmark passports

A passport is a versioned, privacy-filtered JSON record of one measured execution
configuration. Export one with `bench --passport FILE` or one per successfully
measured competitor with `race --passport-dir DIR`.

```bash
llmrig passport verify benchmark.passport.json
```

Verification is offline and read-only. It checks schema, SHA-256 identifiers and
configuration fingerprints, aggregates, impossible states, and known privacy
constraints. These hashes are deterministic identity and integrity checks—not
signatures, independent attestations, benchmark certification, or proof that a
claimed measurement is true. User-supplied native artifact IDs are path-independent
and do not attest file contents.

Passports are `exact` when configuration fingerprints match.
`comparable_with_warnings` means the logical model and workload match but artifact,
format, quantization, runtime, runtime version, or hardware differs.
`not_comparable` means the logical model or workload differs. None of these labels
ranks results. A failed race exports no competitor passports.

## Other commands

Inspect hardware and Ollama readiness:

```bash
llmrig doctor
llmrig doctor --json
```

Check three-state compatibility for a curated ID or exact Hugging Face repository:

```bash
llmrig can qwen3.8:27b-mlx
llmrig can owner/repository --json
```

`can` exits `0` for compatible, `1` for incompatible, and `2` for unknown or
unresolvable. Memory fit is a conservative planning estimate, not a prediction of
performance.

Inspect the Qwen-first catalog:

```bash
llmrig models --offline --fit
llmrig models --fit
llmrig models --refresh --fit
llmrig models --all --fit
```

The curated snapshot is the only layer eligible for automatic setup. Live discovery
is informational and cannot establish package size, fit, runtime compatibility, or
installation trust from a repository name.

Use the curated recommendation/setup workflow when desired:

```bash
llmrig recommend --category official --preference balanced
llmrig setup --model qwen3.8:27b-mlx --context 32768
```

`setup` is mutating: after confirmation it may pull a curated Ollama artifact and
run the full benchmark. Running bare `llmrig` starts the interactive version of that
workflow. LLMRig does not install Ollama itself.

Run offline project checks, or explicitly include live discovery:

```bash
llmrig check
llmrig check --online
```

## Evidence and scope

LLMRig models compatibility as:

```text
hardware × model × artifact × quantization × runtime × context × measured performance
```

Important boundaries:

- verified, inferred, measured, estimated, and unknown are not interchangeable
- discovery does not establish installation trust
- local does not imply executable, measurable, measured, or recommended
- benchmark evidence outranks planning heuristics
- throughput does not establish model quality
- there is no universal model/runtime score
- prediction and prediction-versus-measurement calibration are not implemented yet

Official curated entries are upstream Qwen models distributed through the selected
local backend. Community reduced-refusal entries are third-party derivatives. Their
behavior does not imply better reasoning, accuracy, safety, or reliability; review
their provenance and licensing before use.

## Development

```bash
python3 -m compileall -q llmrig.py _llmrig
python3 -m unittest discover -s tests -v
python3 llmrig.py --version
python3 llmrig.py solve --help
python3 llmrig.py check
```

See [CONTRIBUTING.md](https://github.com/sunilteja93/llmrig/blob/main/CONTRIBUTING.md)
for architecture and evidence invariants and
[RELEASING.md](https://github.com/sunilteja93/llmrig/blob/main/RELEASING.md) for the
release sequence.

## Repository layout

```text
llmrig/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── dependabot.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── ci.yml
│       ├── codeql.yml
│       └── publish-to-pypi.yml
├── _llmrig/                 # private implementation package
├── assets/llmrig-terminal.svg
├── tests/
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── RELEASING.md
├── SECURITY.md
├── llmrig.py                # CLI and small public SDK facade
└── pyproject.toml
```

## Roadmap

Future work may include broader model-family coverage, a RigGraph representation,
privacy-preserving community benchmark evidence, prediction and calibration, and
stable runtime adapter/plugin interfaces. These are directions, not current product
claims. New functionality must preserve provenance and must not turn discovery
metadata into automatic installation trust.

## License and references

LLMRig is released under the
[MIT License](https://github.com/sunilteja93/llmrig/blob/main/LICENSE).

- [Qwen on Hugging Face](https://huggingface.co/Qwen)
- [Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)
- [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
- [Ollama documentation](https://docs.ollama.com/)
- [Ollama generate API](https://docs.ollama.com/api/generate)
- [Ollama running-model API](https://docs.ollama.com/api/ps)
