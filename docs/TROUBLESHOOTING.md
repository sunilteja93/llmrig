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

## oMLX start succeeds, but the service never becomes ready

An Autopilot receipt can show artifact acquisition as completed, followed
by a failed start_runtime action. No model has been loaded or measured in
that case, and a Rig Report must not claim verified inference.

LLMRig checks the JSON response from OMLX_BASE_URL/v1/models, defaulting
to loopback port 8000. A successful exit from omlx start is not proof
of readiness. Check the service and bound port before retrying.

~~~bash
command -v omlx
omlx --version
printf 'Configured endpoint: %s\n' "${OMLX_BASE_URL:-http://127.0.0.1:8000}"
curl -sS --max-time 5 -o /dev/null -w 'API HTTP status: %{http_code}\n' "${OMLX_BASE_URL:-http://127.0.0.1:8000}/v1/models"
lsof -nP -iTCP:8000 -sTCP:LISTEN
tail -n 40 "$HOME/.omlx/logs/server.log"
~~~

For Homebrew installs, inspect brew services info omlx and the service logs.
A non-200 HTTP status may indicate API authorization rather than an absent
server. Configure OMLX_API_KEY for LLMRig where authentication is required,
and never paste that key into logs, receipts, or chats.

A stopped service, port conflict, unexpected custom port, and failed API
authorization are different problems. Redact private paths, usernames,
and credentials before sharing service logs.

Progress showing 0.00B does not necessarily mean model acquisition failed.
The files may already be cached locally.
