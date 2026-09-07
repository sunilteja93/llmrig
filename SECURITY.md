# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability that could expose user data, execute unintended code, bypass a trust boundary, or expose a local inference service.

Use GitHub's private vulnerability reporting / Security Advisory workflow if the
repository presents that option. Repository documentation does not assert that the
feature is enabled. Otherwise, use a private contact method published on the
maintainer's GitHub profile and initially share only enough information to establish
contact.

A useful report includes:

- affected LLMRig version or commit
- operating system and Python version
- local inference backend/version
- reproduction steps
- expected and observed behavior
- impact
- a suggested fix, if you have one

## Trust and privacy boundaries

Live discovery is metadata retrieval, not installation trust. A discovered repository
remains informational until its runtime identifier and provenance are reviewed and
added to the curated catalog. Users remain responsible for the trust, license, and
content of models and runtimes they install.

LLMRig invokes local runtime processes and connects to a configurable Ollama service.
The default is loopback. Do not expose an unauthenticated runtime service to an
untrusted network, and treat a non-loopback `--host` or `OLLAMA_HOST` as an explicit
trust decision.

Native GGUF and MLX-LM execution accepts explicit local paths. Retained private
locator state is confined to non-serializable `InventoryTarget` / `ExecutionTarget`
objects and passed to runtimes as argument-list values; public solve JSON, human
output, and sanitized execution failures omit it. A
user-supplied file association and structural check do not independently attest the
artifact's identity, weights, safety, or runtime behavior.

Benchmark passports and public solve results are designed to exclude known private
paths and machine identities, but review any output before sharing it. Hardware facts,
model identifiers, runtime versions, and measurements may still be identifying in a
particular context. Errors crossing public boundaries are categorical and sanitized.

Passport SHA-256 values provide deterministic identity and integrity checks. They are
not signatures, cryptographic attestations of model contents, independent benchmark
verification, or proof that a reported measurement is true. Path-independent native
artifact identifiers can refer to different files on different machines.

## Supported versions

Until LLMRig reaches a stable 1.0 release, security fixes are applied to the latest
release line. Users should update to the newest available version before reporting
an issue that may already be fixed.
