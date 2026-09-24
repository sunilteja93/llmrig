# Runtime Intelligence

`llmrig runtimes` is the read-only view of local inference runtime readiness.

It intentionally answers a narrower question than `solve`:

> Which known runtimes are present on this machine, and what evidence does LLMRig have about their readiness?

It does **not** claim that an installed runtime has a model available, that an artifact is executable, or that any runtime is faster or better.

## Current runtime adapters

- oMLX
- Ollama
- MLX-LM
- llama.cpp

## Evidence boundaries

- Runtime installation is separate from service availability.
- Runtime readiness is separate from model/artifact availability.
- Generic `safetensors` packaging is not proof of MLX/oMLX compatibility.
- CLI paths shown in public JSON must not expose the current user's home-directory identity.
- Probing is observational only: it does not install software, start services, download models, or execute weights.

## Commands

```bash
llmrig runtimes
llmrig runtimes --json
```

The JSON form is schema-versioned so future `solve`, Autopilot, and RigGraph stages can consume the same observations without scraping terminal text.
