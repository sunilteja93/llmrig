# v0.8 draft notes

## Runtime intelligence

- Adds `llmrig runtimes` and `llmrig runtimes --json`.
- Detects oMLX, Ollama, MLX-LM, and llama.cpp through the v0.8 adapter layer.
- Separates installation, service readiness, blockers, unknowns, formats, and execution interface.
- Sanitizes home-directory CLI paths in public runtime JSON.
- Keeps all runtime probing read-only: no installation, service start, model download, or weight execution.

## Still in progress for v0.8

- Surface adapter observations inside `llmrig solve`.
- Add exact Hugging Face artifact-to-runtime candidate construction.
- Add oMLX local inventory and comparable verification measurements.
