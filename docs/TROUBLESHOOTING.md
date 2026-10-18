# Troubleshooting local installation and CLI failures

## Error: uninstall-no-record-file or ModuleNotFoundError: _llmrig

A pre-existing Python environment can have an incomplete install record for
LLMRig. When pip install -e . refuses to uninstall that old distribution,
the llmrig console script might remain broken. This does not demonstrate a
model compatibility failure.

**Do not delete the old environment or model caches as a first step.**
From the LLMRig repository root, create and verify a separate fresh
environment:

~~~bash
deactivate 2>/dev/null || true
python3 -m venv .venv-fresh
source .venv-fresh/bin/activate
python -m pip install -e .
python -c "import _llmrig.cli; print('LLMRig source import OK')"
python -m pip show llmrig
llmrig --help
~~~

To test source code without the package launcher, from the repository root:

~~~bash
python -m _llmrig.cli runtimes
python -m _llmrig.cli doctor mlx-community/Qwen3-0.6B-4bit
~~~

If source invocation works but llmrig fails, inspect what shell and Python
environment are actually being used:

~~~bash
pwd
command -v python
command -v llmrig
python -m pip show llmrig
python -c "import sys; print(sys.executable)"
~~~

A pipx global executable, a different terminal tab, or a stale virtual
environment can shadow a local editable installation.

Do not run llmrig run MODEL until read-only runtimes, doctor, and plan
work consistently. Run may propose model downloads and other mutations,
which require explicit user approval.

### macOS zsh comments

A pasted comment starting with # can trigger command not found when interactive
comments are disabled. Paste command lines without explanatory comment lines.
This warning is independent of the Python installation problem.
