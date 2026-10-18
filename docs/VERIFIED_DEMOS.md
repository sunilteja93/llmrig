# Verified Autopilot field runs

These are **real local LLMRig execution observations**, not simulations, predicted
performance, independent audits, or interchangeable model benchmarks. The
terminal results and browser report screenshots were captured during live
first-run validation on an Apple M4 Max with 48 GiB unified memory
(October 17-18, 2026).

## Run A: oMLX, small Qwen MLX artifact

| Field | Observed |
| --- | --- |
| Model repository | `mlx-community/Qwen3-0.6B-4bit` |
| Artifact revision | `73e3e38d981303bc594367cd910ea6eb48349da8` |
| Runtime | oMLX 0.6.4 |
| Artifact format | MLX, 4-bit |
| Machine | Apple M4 Max, 48 GiB unified memory |
| Context configuration | 4096 tokens |
| Execution receipt | `receipt-38bdf66af4a4a9ce` |
| Verification | `race-v2`, 2 measured runs, completed |
| Reported latency | 0.29 s |
| Generation and prompt throughput | Not recorded in this receipt |

Actual commands used:

```bash
llmrig runtimes
llmrig doctor mlx-community/Qwen3-0.6B-4bit
llmrig plan mlx-community/Qwen3-0.6B-4bit
llmrig run mlx-community/Qwen3-0.6B-4bit
llmrig report receipt-38bdf66af4a4a9ce --json --output rig-report-qwen.json
```

The run required explicit approval for artifact acquisition and model
registration. Initially, oMLX was listening on port 8000 but its
`/v1/models` API returned **HTTP 401** because the LLMRig process had not
been given the existing oMLX API key. Once `OMLX_API_KEY` was supplied
privately in the shell and `llmrig runtimes` showed oMLX **ready**, the
approved workflow acquired the pinned artifact, refreshed the runtime's
model inventory and completed verification. No API keys are contained in
the shareable report. A displayed download progress of `0.00B` did not
invalidate acquisition: weights were reusable from the local cache.

Your setup may select a different path, be blocked, or require permission
before downloading anything. Do not claim a successful run until LLMRig
prints a completed receipt.

## Run B: Ollama, installed 27B model

| Field | Observed |
| --- | --- |
| Exact installed Ollama tag | `qwen3.8:27b-mlx` |
| LLMRig logical family | `qwen3.8` |
| Runtime | Ollama 0.33.3 |
| Machine | Apple M4 Max, 48 GiB unified memory |
| Context configuration | 4096 tokens |
| Verification | `race-v2`, 2 measured runs, completed |
| Generation throughput | **49.93 tokens/s**, runtime-reported |
| Reported latency | 2.5861 s |
| Prompt throughput | Omitted from the corrected report; too-short prompt timing was not a trustworthy throughput denominator |

The corrected report shown in the live viewer used those numbers and
identified the exact Ollama tag. The earlier completed run
(`receipt-72ef6bf78090e3e3`) reported 50.42 generation tokens/s and
2.5619 s, but also reported an implausible multimillion-tokens/s prompt
metric. That earlier prompt measurement **must not** be used as a
benchmark. LLMRig now suppresses prompt-throughput claims when Ollama's
reported prompt phase is shorter than 1 ms. This is a conservative
measurement guard, not proof of caching.

Actual commands used:

```bash
ollama list
ollama show qwen3.8:27b-mlx
llmrig plan qwen3.8:27b-mlx
llmrig run qwen3.8:27b-mlx --context 4096
llmrig report --json --output rig-report-ollama.json
```

This workflow was **verification-only** because the model was already
installed; it did not need to download the 18 GB artifact. The locally
installed model was reported by `ollama show` as 27.8B parameters,
`qwen3_5` architecture and `nvfp4` quantization. At the time of
the demonstration, LLMRig's curated format was labeled `MLX` with
unknown quantization. Treat that **catalog label as unverified**
until the runtime-versus-catalog metadata discrepancy
([issue #63](https://github.com/sunilteja93/llmrig/issues/63)) is resolved.
The run's success does not attest to that catalog classification.

## Open your own real Rig Report

```bash
llmrig report --json --output rig-report.json
```

Open `docs/rig-report-viewer.html` from a local checkout, select your
JSON file, and click **View report**. It will render entirely in your
browser without uploading the report. The report intentionally excludes
filesystem paths, API keys, raw endpoints and historical hardware fields
not captured by the receipt. You may also run `llmrig report --share`
for a Markdown summary. Reports never overwrite an existing file.

For a real screenshot, use the collapsed **Load or change report**
view after pressing **View report**. The diagram in
[`assets/verified-dual-runtime.svg`](../assets/verified-dual-runtime.svg)
is an **illustration reconstructed from the observed report values**,
not a claim that the image itself is cryptographically attested.

## Reproduce responsibly

1. Follow the [installation guide](INSTALL.md) and confirm that
   `llmrig --help` includes `runtimes`, `doctor`, `plan`, `run`
   and `report`.
2. Start with `llmrig runtimes`, `llmrig doctor MODEL` and
   `llmrig plan MODEL`. These steps are read-only.
3. Run only a supported path, review any actions requiring approval,
   and keep local credentials private.
4. Generate a report from the **actual completed receipt**, inspect
   the values and annotate omitted metrics and artifact identity.
5. Report blockages honestly. The 0.6B and 27B runs differ in model,
   format, runtime, prompt-processing detail and possibly caching.
   **Their latency and generation speeds are not a controlled runtime
   comparison.** A valid comparison must use the same artifact,
   prompt, context, versioned workload and equivalent measurement conditions.

CI verifies Python packaging and deterministic behavior on supported
platforms; it does **not** reproduce these physical model executions.
