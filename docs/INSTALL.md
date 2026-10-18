# Install LLMRig: start clean, validate the CLI

LLMRig is a Python 3.9+ command-line tool. It **does not install** Ollama,
oMLX, MLX-LM or llama.cpp just because you installed the CLI.

**Published release: v0.10.0.** Use the stable PyPI package for
the demonstrated `runtimes`, model-specific `doctor`, `plan`,
`run` and `report` flows. If the new version has not propagated
to your package index yet, check
[the GitHub release](https://github.com/sunilteja93/llmrig/releases/tag/v0.10.0)
before retrying. Installation does not start or install model runtimes.

## Recommended: isolated pipx installation

Prerequisites: Python 3.9+, and
[`pipx`](https://pipx.pypa.io/). On macOS with Homebrew:

```bash
brew install pipx
pipx ensurepath
```

Open a new terminal after `pipx ensurepath`, then:

```bash
pipx install llmrig
rehash
llmrig --version
llmrig --help
llmrig runtimes
```

To upgrade an existing installation:

```bash
pipx upgrade llmrig
```

Inspect `llmrig --help` for `runtimes`, `doctor`, `plan`, `run`,
`verify` and `report`. Probe commands must not download model weights.

For development from unreleased source, install an exact Git commit:
`pipx install 'git+https://github.com/sunilteja93/llmrig.git@COMMIT_SHA'`,
substituting the desired commit. Avoid installing an unpinned moving
branch when a reproducible versioned release is available.

## Alternative: clean virtual environment

Use a separate directory **outside the repository** so that Python cannot
mistake the current working directory for a correctly installed package.

macOS / Linux:

```bash
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/llmrig-demo"
"$HOME/.venvs/llmrig-demo/bin/python" -m pip install llmrig
(cd /tmp && "$HOME/.venvs/llmrig-demo/bin/llmrig" --help)
(cd /tmp && "$HOME/.venvs/llmrig-demo/bin/llmrig" runtimes)
```

Windows PowerShell (Git and Python must already be installed):

```powershell
py -m venv "$HOME\llmrig-demo"
& "$HOME\llmrig-demo\Scripts\python.exe" -m pip install "llmrig"
Set-Location $env:TEMP
& "$HOME\llmrig-demo\Scripts\llmrig.exe" --help
& "$HOME\llmrig-demo\Scripts\llmrig.exe" runtimes
```

For development from an existing clone, use a fresh virtual environment
and `python -m pip install -e .`. CI tests this editable workflow on
macOS and Linux, including invoking the CLI **outside** the checkout.
Do not infer that editable installation succeeded solely from
`python -m _llmrig.cli` when you are inside the source directory.

## PyPI release vs GitHub source

The documented first-run workflow targets the published v0.10.0 CLI.
Pin `llmrig==0.10.0` in reproducible Python environments, or use the
exact Git tag/commit if testing unreleased changes.
Do not assume a model runtime is installed just because the CLI is available.

## First read-only check

```bash
llmrig runtimes
llmrig doctor mlx-community/Qwen3-0.6B-4bit
llmrig plan mlx-community/Qwen3-0.6B-4bit
```

`doctor` and `plan` only analyze evidence. `run` may acquire weights,
configure or start a runtime, or load a model, with explicit approval
where mutations are planned. A plan that selects a runnable local model
can verify it without downloading anything, but inference may load it
into memory.

A small model verified on oMLX in
[the field run](VERIFIED_DEMOS.md) required oMLX to be installed and
authenticated. If oMLX answers HTTP **401**, do not disable its security
or paste keys into issue reports. Set `OMLX_API_KEY` privately in the
terminal process and confirm `llmrig runtimes` reports `ready` before
running. The [troubleshooting guide](TROUBLESHOOTING.md) has diagnostics.

## Installation checks and recovery

```bash
command -v llmrig
type -a llmrig
rehash
llmrig --help
```

On macOS zsh, `rehash` refreshes previously cached executable paths.
If `llmrig` throws `ModuleNotFoundError: No module named '_llmrig'`,
verify the interpreter behind its shebang and test the installed CLI
from **outside** the repository. The clean non-editable venv above is
a supported isolation strategy. Avoid deleting model caches or changing
runtime configs to fix a Python import issue.

Never paste API keys, raw server logs containing secrets, or private
filenames into shared reports. `llmrig report` generates a filtered
local artifact and never uploads it automatically. It will not
overwrite an existing output filename.

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
