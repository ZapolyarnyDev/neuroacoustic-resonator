# Neuroacoustic Resonator

English · [Русский](README.ru.md) · [العربية](README.ar.md) · [עברית](README.he.md)

[![CI](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml/badge.svg)](https://github.com/ZapolyarnyDev/neuroacoustic-resonator/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental field of coupled phase oscillators with local metabolism and
plasticity, driven by audio. Field states can be recorded as Sound Protocol v0
streams and rendered as sound. It is not a language model and does not currently
demonstrate language, cognition, or learned acoustic concepts.

## Current status

In the current field the input signal does not reach the output region. Output
responses to every stimulus are identical to the silence control, and the
disturbance decays by a factor of 50–100 per column. The earlier pilot result
(`0.542` cross-seed balanced accuracy, tag `v0-baseline`) came from audio driving
the association region directly; without that drive both the field and an
uncoupled field show no signal ([report](experiments/reports/2026-11-06-9748330.md)).

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/).

```bash
just sync
just check
just preview
```

Every recipe calls `uv run neuroacoustic <command>`; run
`uv run neuroacoustic --help` for the full list.

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

### Research

| Command | Purpose |
|---|---|
| `just report` | Write the key report for the current commit to `experiments/reports/`. |
| `just causal-run` | Run the paired stimulus/control corpus (30 branches from shared equilibrated checkpoints) and rebuild the evidence. |
| `just causal-analyze` | Recalculate checkpoint-aware output and regional causal statistics without simulations. |
| `just causal-parts` | Measure the response with each field component disabled and with the output region cut off. |
| `just causal-ablations` | Run the four development-seed equilibration variants and compare output responses. |
| `just causal-ablations-report` | Rebuild the ablation comparison from completed variants. |

### Field, state and performance

| Command | Purpose |
|---|---|
| `just preview` | Run a short field simulation and write `outputs/field-preview.png`. |
| `just checkpoint [steps] [output]` | Save a long-running field state; defaults to 10,000 steps. |
| `just resume <checkpoint> <steps> <output>` | Continue a saved simulation into a new checkpoint. |
| `just benchmark` | Benchmark field stepping at sizes 64, 128, and 200. |

### Sound Protocol

| Command | Purpose |
|---|---|
| `just protocol-record [config] [steps] [output]` | Record a strict Sound Protocol JSONL stream; defaults to `field_only.yaml`, 128 steps. |
| `just protocol-replay [input] [output]` | Replay JSONL without the field and render the diagnostic WAV. |

### Audio interaction

| Command | Purpose |
|---|---|
| `just audio-devices` | List microphone and output devices. |
| `just live` | Start turn-based microphone interaction and record the session. |
| `just conversation <input.wav> [output.wav]` | Run one recorded WAV through the field and save its response. |

## License

[MIT](LICENSE)
