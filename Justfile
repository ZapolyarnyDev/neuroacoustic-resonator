set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    just --list

# Environment and quality
sync:
    uv sync --locked --dev

check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run pytest

fmt:
    uv run ruff check . --fix
    uv run ruff format .

test *args:
    uv run pytest {{args}}

build:
    uv build

hooks-install:
    uv run pre-commit install

hooks:
    uv run pre-commit run --all-files

# Current research workflow
causal-run:
    uv run python scripts/run_controlled_equilibration.py

causal-analyze:
    uv run python scripts/run_controlled_equilibration.py --analysis-only

causal-ablations:
    uv run python scripts/run_equilibration_ablations.py

causal-ablations-report:
    uv run python scripts/run_equilibration_ablations.py --report-only

# Field and Sound Protocol
preview:
    uv run python main.py

protocol-record config="configs/field_only.yaml" steps="128" output="outputs/protocol/recording.jsonl":
    uv run neuroacoustic-protocol record --config "{{config}}" --steps {{steps}} --output "{{output}}" --summary "{{output}}.summary.json"

protocol-replay input="outputs/protocol/recording.jsonl" output="outputs/protocol/replay.wav":
    uv run neuroacoustic-protocol replay --input "{{input}}" --summary "{{input}}.summary.json" --output-wav "{{output}}"

# Audio interaction
audio-devices:
    uv run python scripts/run_live_conversation.py --list-devices

live:
    uv run python scripts/run_live_conversation.py --config configs/field_only.yaml --input-assoc-gain 0.8 --input-output-gain 0.0 --print-rms --record-dir experiments/audio/live

conversation input output="experiments/audio/voice-conversation.wav":
    uv run python scripts/run_conversation.py --config configs/field_only.yaml --inputs "{{input}}" --output "{{output}}" --summary "{{output}}.summary.json" --input-assoc-gain 0.8 --input-output-gain 0.0

# State and performance utilities
checkpoint steps="10000" output="experiments/states/default-10k.npz":
    uv run python scripts/save_checkpoint.py --config configs/default.yaml --steps {{steps}} --output "{{output}}"

resume checkpoint steps output:
    uv run python scripts/resume_checkpoint.py --checkpoint "{{checkpoint}}" --steps {{steps}} --output "{{output}}"

benchmark:
    uv run python scripts/benchmark_field.py --sizes 64,128,200 --steps 1000 --repeats 3 --output outputs/benchmarks/field-step.csv
