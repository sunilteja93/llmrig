# Install LLMRig: start clean, validate the CLI

LLMRig is a Python 3.9+ command-line tool. It **does not install** Ollama,
oMLX, MLX-LM or llama.cpp just because you installed the CLI.

**Release distinction (October 2026):** PyPI currently has **0.9.3**.
The latest `main` branch contains the newer `runtimes`, model-specific
`doctor`, `plan`, `run` and `report` facelift work. Before the next
PyPI release, use the **GitHub source install below** to reproduce the
[verified field runs](VERIFIED_DEMOS.md). `llmrig --version` may still
print `0.9.3` for unreleased source changes: the version string alone
does not prove that a command is available.

## Recommended: isolated pipx install of current GitHub main

Prerequisites: Git, a Python 3.9+ interpreter and
[`pipx`](https://pipx.pypa.io/). macOS with Homebrew:

```bash
brew install pipx
pipx ensurepath
```

Open a **new terminal** after `pipx ensurepath`, then:

```bash
pipx install 'git+https://github.com/sunilteja93/llmrig.git@main'
rehash
llmrig --help
llmrig runtimes
```

Verify that `llmrig --help` offers `runtimes`, `doctor`, `plan`,
`run`, `verify` and `report`. The runtime scan must work without
downloading a model or starting a service.

`@main` tracks an **unreleased moving branch**. For reproducibility,
replace `main` in the URL with an exact Git commit SHA from the
[repository history](https://github.com/sunilteja93/llmrig/commits/main/).
A new PyPI release should replace this interim recommendation after
release-artifact smoke testing.

To switch an existing isolated pipx installation to the current GitHub
source, use `pipx install --force 'git+https://github.com/sunilteja93/llmrig.git@main'`.
Do not confuse a global `~/.local/bin/llmrig` with an activated virtual
environment's `bin/llmrig`.

## Alternative: clean virtual environment

Use a separate directory **outside the repository** so that Python cannot
mistake the current working directory for a correctly installed package.

macOS / Linux:

```bash
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/llmrig-demo"
"$HOME/.venvs/llmrig-demo/bin/python" -m pip install 'git+https://github.com/sunilteja93/llmrig.git@main'
(cd /tmp && "$HOME/.venvs/llmrig-demo/bin/llmrig" --help)
(cd /tmp && "$HOME/.venvs/llmrig-demo/bin/llmrig" runtimes)
```

Windows PowerShell (Git and Python must already be installed):

```powershell
py -m venv "$HOME\llmrig-demo"
& "$HOME\llmrig-demo\Scripts\python.exe" -m pip install "git+https://github.com/sunilteja93/llmrig.git@main"
Set-Location $env:TEMP
& "$HOME\llmrig-demo\Scripts\llmrig.exe" --help
& "$HOME\llmrig-demo\Scripts\llmrig.exe" runtimes
```

For development from an existing clone, use a fresh virtual environment
and `python -m pip install -e .`. CI tests this editable workflow on
macOS and Linux, including invoking the CLI **outside** the checkout.
Do not infer that editable installation succeeded solely from
`python -m _llmrig.cli` when you are inside the source directory.

## PyPI release vs GitHub main

For the latest **published** package, rather than the unreleased
walkthrough above:

```bash
pipx install llmrig
llmrig --help
```

The published command set may differ from the latest GitHub docs until
the next release is uploaded. **Check the actual help output** before
following any Autopilot workflow. Do not call a feature shipped simply
because code exists on `main`.

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
