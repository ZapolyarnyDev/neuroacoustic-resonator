set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

[private]
default:
    just --list

[group('environment')]
sync:
    uv sync --locked --dev

[group('environment')]
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run pytest

[group('environment')]
fmt:
    uv run ruff check . --fix
    uv run ruff format .

[group('environment')]
test *args:
    uv run pytest {{args}}

[group('environment')]
build:
    uv build

[group('environment')]
hooks-install:
    uv run pre-commit install

[group('environment')]
hooks:
    uv run pre-commit run --all-files

[group('research')]
report:
    uv run neuroacoustic report

[group('research')]
causal-run:
    uv run neuroacoustic causal

[group('research')]
causal-analyze:
    uv run neuroacoustic causal --analysis-only

[group('research')]
causal-parts:
    uv run neuroacoustic parts

[group('research')]
causal-ablations:
    uv run neuroacoustic ablations

[group('research')]
causal-ablations-report:
    uv run neuroacoustic ablations --report-only

[group('field')]
preview:
    uv run neuroacoustic preview

[group('field')]
checkpoint steps="10000" output="experiments/states/default-10k.npz":
    uv run neuroacoustic checkpoint --config configs/default.yaml --steps {{steps}} --output "{{output}}"

[group('field')]
resume checkpoint steps output:
    uv run neuroacoustic resume --checkpoint "{{checkpoint}}" --steps {{steps}} --output "{{output}}"

[group('field')]
benchmark:
    uv run neuroacoustic benchmark --sizes 64,128,200 --steps 1000 --repeats 3 --output outputs/benchmarks/field-step.csv

[group('protocol')]
protocol-record config="configs/field_only.yaml" steps="128" output="outputs/protocol/recording.jsonl":
    uv run neuroacoustic protocol record --config "{{config}}" --steps {{steps}} --output "{{output}}" --summary "{{output}}.summary.json"

[group('protocol')]
protocol-replay input="outputs/protocol/recording.jsonl" output="outputs/protocol/replay.wav":
    uv run neuroacoustic protocol replay --input "{{input}}" --summary "{{input}}.summary.json" --output-wav "{{output}}"

[group('audio')]
audio-devices:
    uv run neuroacoustic live --list-devices

[group('audio')]
live:
    uv run neuroacoustic live --config configs/field_only.yaml --input-output-gain 0.0 --print-rms --record-dir experiments/audio/live

[group('audio')]
conversation input output="experiments/audio/voice-conversation.wav":
    uv run neuroacoustic conversation --config configs/field_only.yaml --inputs "{{input}}" --output "{{output}}" --summary "{{output}}.summary.json" --input-output-gain 0.0
