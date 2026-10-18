# Live validation matrix

This document separates **existing observed measurements** from **future live
verification work**. A test case is not a claim of successful execution.

| Scenario | Target | Current public evidence | Next validation |
| --- | --- | --- | --- |
| oMLX small-model Autopilot | M4 Max 48 GiB, mlx-community/Qwen3-0.6B-4bit | **Completed live** with exact HF revision, explicit approval, 2 verification runs, 0.29 s recorded latency | Independent user reproduction; collect complete throughput only if runtime reports trustworthy metrics |
| Ollama installed 27B Autopilot | M4 Max 48 GiB, qwen3.8:27b-mlx | **Completed live** with 2 runs, corrected report 49.93 generation tok/s, 2.5861 s recorded latency; no prompt TPS claim | Reconcile runtime quantization vs curated format (#63), independent tester and repeat runs |
| Earlier MLX-LM benchmark | Apple Silicon, MLX-LM, mlx-community/Qwen3-0.6B-4bit | Two race-v2 measurements in README; distinct from oMLX Autopilot run | Verify same model/runtime/workload under explicit controlled conditions |
| Mid-size Hugging Face Qwen setup | Apple M4 Max 48 GiB, mlx-community/Qwen3.5-27B-4bit | Planning example only, **not** an end-to-end success claim | Execute and record revision, memory headroom, runtime behavior and verified metrics |
| Large GGUF failure diagnosis | Apple M4 Max 48 GiB, unsloth/Beam-1.0-501B-A23B-GGUF | v0.9.3 adds aggregate split-shard weight fit reasoning | Confirm read-only doctor and plan explain model-size incompatibility without pretending success |
| Cross-platform user journey | Volunteer Linux/Windows machines | CI platform coverage is not a proof of live user setup | Have an independent contributor run it and file a redacted receipt |

Detailed receipts, reproduction commands, caveats and report creation:
[VERIFIED_DEMOS.md](VERIFIED_DEMOS.md). CI does not verify physical runtime inference.

## Acceptance criteria

- A new user understands why each candidate is viable, blocked, or unknown.
- Read-only plan and doctor never download weights or start services.
- Any artifact download, installation or runtime mutation requires explicit intent.
- No incompatibility, missing provenance, or unknown becomes a successful plan.
- Measured values come from executed verification and are not fabricated.
- A generated share report contains no local filesystem path, username, secret,
  raw endpoint, or unsupported historical hardware claim.
- The report file is not uploaded, and never silently overwrites an existing file.
- CI passes on the supported Python / OS matrix; live setups are separately marked
  as verified or unverified.
- At least five independent testers attempt a complete setup and report where
  they stopped, including no-success attempts.

## Field-test log template

| Date | Tester / platform (public only) | Exact model revision | Runtime version | Plan result | Apply result | Verify result | Evidence / issue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unverified | Open | Open | Open | Pending | Pending | Pending | Do not populate with invented values |

## Distribution backlog

1. Publish a short screen recording of a real successful run, with receipt and
   proof that no setup actions were taken before approval.
2. **Complete:** a static offline Rig Report viewer is included; check its
   missing-metric disclosure and private-field boundary in every release.
3. Validate downloadable self-contained distributions for macOS/Linux/Windows;
   then add a Homebrew tap or documented package-manager integration. Do not
   advertise an installer until it exists and smoke tests pass.
4. Publish at least three fully reproducible success and failure stories with
   clear provenance.
5. Avoid ranking throughput against model quality and avoid publishing community
   measurements without participant consent.
