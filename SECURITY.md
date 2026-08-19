# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability that could expose user data, execute unintended code, bypass a trust boundary, or expose a local inference service.

Use GitHub's private vulnerability reporting / Security Advisory workflow when it is available for the repository. If private reporting is not available, contact a maintainer privately through their GitHub profile and provide only the minimum information needed to establish contact.

A useful report includes:

- affected LLMRig version or commit
- operating system and Python version
- local inference backend/version
- reproduction steps
- expected and observed behavior
- impact
- a suggested fix, if you have one

## Security design principles

LLMRig intentionally separates live model discovery from automatic installation. A repository found through live discovery is informational until its local-backend identifier and provenance are reviewed and added to the curated catalog.

LLMRig also assumes Ollama is a **localhost service**. Do not expose an unauthenticated Ollama endpoint directly to the public internet.

Benchmark reports should be reviewed before publication even though LLMRig removes known local path fields from its shareable benchmark metadata.

## Supported versions

Until LLMRig reaches a stable 1.0 release, security fixes are applied to the latest release line. Users should update to the newest available version before reporting an issue that may already be fixed.
