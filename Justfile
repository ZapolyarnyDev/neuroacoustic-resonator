set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    just --list

sync:
    uv sync --locked --dev

run:
    uv run python main.py

preview:
    uv run python main.py

metrics:
    uv run python scripts/run_long_metrics.py --config configs/default.yaml --steps 10000 --sample-interval 10 --output outputs/metrics/default-10k.csv

metrics-long:
    uv run python scripts/run_long_metrics.py --config configs/long_run.yaml --steps 10000 --sample-interval 10 --output outputs/metrics/long-run-10k.csv

metrics-custom config steps sample_interval output:
    uv run python scripts/run_long_metrics.py --config {{config}} --steps {{steps}} --sample-interval {{sample_interval}} --output {{output}}

benchmark:
    uv run python scripts/benchmark_field.py --sizes 64,128,200 --steps 1000 --repeats 3 --output outputs/benchmarks/field-step.csv

benchmark-custom sizes steps repeats output:
    uv run python scripts/benchmark_field.py --sizes {{sizes}} --steps {{steps}} --repeats {{repeats}} --output {{output}}

audio:
    uv run python scripts/render_audio_demo.py --config configs/default.yaml --duration-seconds 5 --output experiments/audio/default-demo.wav

audio-demo:
    uv run python scripts/render_audio_demo.py --config configs/audio_demo.yaml --duration-seconds 10 --output experiments/audio/audio-demo.wav

audio-custom config duration output:
    uv run python scripts/render_audio_demo.py --config {{config}} --duration-seconds {{duration}} --output {{output}}

audio-steps steps:
    uv run python scripts/render_audio_demo.py --config configs/default.yaml --steps {{steps}} --output experiments/audio/default-{{steps}}-steps.wav

live:
    uv run python scripts/live_field.py --config configs/default.yaml --interval-ms 30 --steps-per-update 1

live-input:
    uv run python scripts/live_field.py --config configs/synthetic_input.yaml --interval-ms 30 --steps-per-update 2

live-field:
    uv run python scripts/live_field.py --config configs/field_only.yaml --interval-ms 30 --steps-per-update 2

live-custom config interval_ms steps_per_update:
    uv run python scripts/live_field.py --config {{config}} --interval-ms {{interval_ms}} --steps-per-update {{steps_per_update}}

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

typecheck:
    uv run mypy .

test:
    uv run pytest

check: lint typecheck test

hooks-install:
    uv run pre-commit install

hooks:
    uv run pre-commit run --all-files
