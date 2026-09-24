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

A PEP 660 editable install (`python -m pip install -e .`) is also exercised by CI,
but a normal local install is the conservative default when a platform-specific
editable-install environment behaves differently.

Confirm that the active command comes from the repository environment:

```bash
which python
which llmrig
llmrig --version
```

`which llmrig` should point inside `.venv/bin/` while the environment is active.
Do not use `--break-system-packages` for normal LLMRig development.

For the v0.8 runtime-intelligence work, useful smoke tests are:

```bash
llmrig runtimes
llmrig runtimes --json
llmrig solve mlx-community/Qwen3.5-27B-4bit --json
```

The runtime probe and default solve path are observational only. They do not install
runtimes, start services, download model weights, or execute inference. The solve
smoke test resolves Hugging Face metadata only; an oMLX/MLX runtime candidate does
not imply that the model is already local or measurable.

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
