# LLMRig development setup

Use an isolated virtual environment when developing LLMRig from a source checkout.
This is the recommended path on Homebrew-managed macOS Python installations and
avoids PEP 668 `externally-managed-environment` errors.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Confirm that the active command comes from the repository environment:

```bash
which python
which llmrig
llmrig --version
```

`which llmrig` should point inside `.venv/bin/` while the environment is active.
Do not use `--break-system-packages` for normal LLMRig development.

A normal local install is the default contributor path because it exercises the
same wheel packaging boundary users receive. After pulling source changes, rerun
`python -m pip install .` before testing the installed command.

Editable installs are useful during active implementation but are validated
separately because PEP 660 behavior can vary across Python/pip/setuptools versions.
If using editable mode, verify it from outside the repository before relying on it:

```bash
python -m pip install -e .
cd ..
python -c "import llmrig, _llmrig"
cd llmrig
```

For the v0.8 runtime-intelligence work, a useful smoke test is:

```bash
llmrig runtimes
llmrig runtimes --json
```

The runtime probe is observational only. It does not install runtimes, start
services, download models, or execute model weights.

Run the project checks before pushing changes:

```bash
python -m compileall -q llmrig.py _llmrig
python -m unittest discover -s tests -v
llmrig check
git diff --check
```

Leave the development environment with:

```bash
deactivate
```
