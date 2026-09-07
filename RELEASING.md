# Releasing LLMRig

LLMRig publishes to PyPI from a published GitHub Release through Trusted Publishing
(OIDC). No long-lived PyPI token is part of the workflow.

1. Open a focused release PR that updates `llmrig.VERSION`, `CHANGELOG.md`,
   `CITATION.cff`, user documentation, and release-only tests.
2. Run the complete cross-platform CI matrix and the release-artifact validation job.
3. Confirm the wheel and sdist contain `llmrig.py`, `_llmrig`, metadata, and the
   console entry point; install both in clean environments and run read-only smokes.
4. Confirm version, changelog date, citation metadata, supported Python versions,
   and product claims agree, then merge the reviewed release PR.
5. Create and publish a GitHub Release tagged exactly `v` plus `llmrig.VERSION`, from
   the intended release commit. Do not reuse or move an existing release tag.
6. Let `publish-to-pypi.yml` check out the tag, reject a tag/source version mismatch
   or a tagged commit not contained in `main`, build and smoke-test the artifacts,
   and publish with OIDC.
7. Verify the version and metadata on PyPI and compare the published artifact hashes
   with the workflow artifacts.
8. In a new clean environment, install from PyPI and verify `llmrig --version`,
   `import llmrig`, `import _llmrig`, and a read-only `llmrig.solve(...)` smoke.

If publication fails, fix the cause and rerun the failed release workflow. Do not
create a second tag for the same version or use a manual token-based publication path.
