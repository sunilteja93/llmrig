# First run: one verifiable local model

LLMRig is an evidence-based execution planner, not a model directory and not a
replacement for Ollama, oMLX, MLX-LM or llama.cpp. It will not silently guess
a working installation from missing evidence.

## 1. Inspect your actual machine

~~~sh
pipx install llmrig
llmrig --version
llmrig runtimes
~~~

If a runtime is missing or stopped, fix its installation and readiness using
its **official** instructions, then rerun the probe. Runtime detection is not a
promise of execution support.

## 2. Diagnose an exact model

~~~sh
llmrig doctor mlx-community/Qwen3.5-27B-4bit
llmrig plan mlx-community/Qwen3.5-27B-4bit
~~~

The first command is a brief read-only diagnosis; the second shows the full
candidate evidence, possible actions, and blockers. A model can fit in memory
and still be blocked because its runtime, provenance, or artifact format is not
supported.

Use a smaller model, a different evidenced format, or a ready runtime if
diagnostics indicate that the original configuration is not viable. Do not
assume a model's file naming proves it can be executed.

## 3. Approve the supported plan

~~~sh
llmrig run mlx-community/Qwen3.5-27B-4bit
~~~

You see the proposed changes before any mutation. An interactive confirmation
or explicit non-interactive approval is required. If no supported plan exists,
LLMRig stops. A blocked run is not a successful demo.

## 4. Verify and share an actual result

After a successful run with measured verification:

~~~sh
llmrig verify
llmrig report --share --output rig-report.md
llmrig report --json --output rig-report.json
~~~

A report is generated locally only. It includes only whitelisted public receipt
facts such as exact model identity, artifact revision if recorded, runtime,
verification method, and measured values if present. It intentionally excludes
filesystem paths, endpoint addresses, secrets, and hardware labels not saved in
the receipt.

The generated file is never uploaded or overwritten automatically. Inspect
its contents yourself before posting it. Receipts and reports are
self-reported, not independently attested.

## 5. Share a reproducible result

Useful bug reports and success stories specify the exact model revision,
runtime and version, OS and architecture, configured context, LLMRig version,
whether the model was already cached, and how the measurement was produced.
Share only public hardware descriptors and scrub any machine-specific identifiers.

See [validation matrix](https://github.com/sunilteja93/llmrig/blob/main/docs/VALIDATION_MATRIX.md).
