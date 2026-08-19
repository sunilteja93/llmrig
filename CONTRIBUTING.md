# Contributing to LLMRig

Thanks for helping improve LLMRig. The project is intended to be community-driven, and contributions of code, tests, documentation, hardware results, and verified model metadata are welcome.

## Before you start

Please keep these principles in mind:

1. **Prefer verified facts over guesses.** Model sizes, context limits, backend support, and model identifiers should come from upstream or primary sources.
2. **Discovery is not trust.** A model found through live search must not automatically become installable.
3. **Keep hardware recommendations conservative.** Fitting weights into memory is not the same as providing a usable local experience.
4. **Keep benchmarks reproducible.** Model, quantization, context, runtime version, and test settings should be explicit.
5. **Keep platform claims evidence-based.** If a path is not tested on real hardware, say so.

## Ways to contribute

Useful contributions include:

- support for additional model families
- support for additional local inference backends
- verified Ollama model identifiers and aliases
- Apple Silicon, NVIDIA, AMD, and Intel hardware detection
- Windows and Linux hardware fixes
- model-fit and context heuristics
- benchmark improvements
- documentation and examples
- unit tests and regression tests
- issue triage and reproducible bug reports

## Development setup

LLMRig currently uses only the Python standard library.

After cloning your fork locally:

```bash
cd llmrig
python3 -m py_compile llmrig.py
python3 -m unittest discover -s tests -v
python3 llmrig.py check
```

With internet access:

```bash
python3 llmrig.py check --online
```

## Pull-request workflow

1. Fork the repository.
2. Create a focused branch, for example `feature/amd-vram-detection`.
3. Make the smallest coherent change that solves the problem.
4. Add or update tests for behavior changes.
5. Run the local validation commands above.
6. Update documentation when user-visible behavior changes.
7. Open a pull request and explain what changed, why, and how it was tested.

## Adding a curated model

A curated model can be auto-pulled, so the bar is higher than for discovery-only results.

A model contribution should include:

- upstream model name and publisher
- exact local-backend identifier/tag
- verified package/download size
- quantization/precision
- supported platform/backend
- advertised maximum context
- input modalities
- source links used for verification
- whether the model is official or a community derivative
- at least one test or validation path when practical

Do not mark a newly discovered third-party repository as auto-installable solely because its name looks correct.

## Adding a new model family

LLMRig is Qwen-first today, but new model families are welcome. Prefer changes that keep the generic workflow intact:

```text
hardware → discover → recommend → setup → benchmark
```

A new family should ideally have a clear discovery source, a curated local-ready layer, and tests that prevent one family from breaking another.

## Benchmark changes

Benchmark changes should preserve comparability. If you change a prompt, timing method, warm-up behavior, context allocation, or correctness test, explain why and update tests/documentation.

Never commit benchmark output that contains private paths, tokens, secrets, usernames, or other sensitive machine metadata.

## Style

- Keep the CLI dependency-free unless a dependency provides clear value that cannot reasonably be achieved with the standard library.
- Prefer readable Python over clever Python.
- Keep functions focused and testable.
- Avoid silently changing existing CLI behavior.
- Add comments where platform-specific behavior would otherwise be surprising.

## Questions and ideas

Open a feature-request issue, or a GitHub Discussion if Discussions are enabled, before investing in a large implementation. That gives maintainers and contributors a chance to agree on direction first.
