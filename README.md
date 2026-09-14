# Neuroacoustic Resonator

[Русская версия](README.ru.md)

[![CI](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml/badge.svg)](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental field of coupled oscillators that transforms audio into evolving
dynamics, records them as Sound Protocol v0, and renders protocol-driven responses.
It is not a language model and does not currently demonstrate language, cognition,
or learned acoustic concepts.

## Current status

The active research question is whether different sounds produce reproducible,
seed-invariant field responses. Paired stimulus/control branches now start from the
same equilibrated checkpoint. The first 3-root pilot reached `0.542` cross-seed
balanced accuracy against `0.25` chance, with checkpoint-clustered 95% CI
`[0.274, 0.601]`. This is promising pilot evidence, not a completed result.

Current pipeline:

```text
audio -> input/association/output field -> Sound Protocol v0 -> analysis or renderer
```

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/).

```bash
just sync
just check
just preview
```

Generated artifacts are written to `experiments/` and `outputs/` and are not
committed by default.

## Commands

Run `just` to list the current public recipes.

### Environment and quality

| Command | Purpose |
|---|---|
| `just sync` | Install the locked runtime and development environment. |
| `just check` | Run Ruff, formatting validation, mypy, and the complete test suite. |
| `just fmt` | Apply safe Ruff fixes and format Python files. |
| `just test [pytest args]` | Run all tests or pass a path/options to pytest. |
| `just build` | Build the Python package. |
| `just hooks-install` | Install the repository pre-commit hook. |
| `just hooks` | Run all pre-commit checks immediately. |

### Active research

| Command | Purpose |
|---|---|
| `just causal-run` | Run the 30-branch equilibrated stimulus/control pilot and rebuild all evidence. |
| `just causal-analyze` | Recalculate checkpoint-aware output and regional causal statistics without simulations. |

### Field and Sound Protocol

| Command | Purpose |
|---|---|
| `just preview` | Run a short field simulation and write `outputs/field-preview.png`. |
| `just protocol-record [config] [steps] [output]` | Record a strict Sound Protocol JSONL stream; defaults to `field_only.yaml`, 128 steps. |
| `just protocol-replay [input] [output]` | Replay JSONL without the field and render the diagnostic WAV. |

### Audio interaction

| Command | Purpose |
|---|---|
| `just audio-devices` | List microphone and output devices. |
| `just live` | Start turn-based microphone interaction and record the session. |
| `just conversation <input.wav> [output.wav]` | Run one recorded WAV through the field and save its response. |

### State and performance

| Command | Purpose |
|---|---|
| `just checkpoint [steps] [output]` | Save a long-running field state; defaults to 10,000 steps. |
| `just resume <checkpoint> <steps> <output>` | Continue a saved simulation into a new checkpoint. |
| `just benchmark` | Benchmark field stepping at sizes 64, 128, and 200. |

## License

[MIT](LICENSE)
