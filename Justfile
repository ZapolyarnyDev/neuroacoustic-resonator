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

experiments:
    uv run python scripts/run_experiments.py --config configs/synthetic_input.yaml --output-dir experiments/logs --propagation-horizon 1024

experiments-custom config output_dir propagation_horizon:
    uv run python scripts/run_experiments.py --config {{config}} --output-dir {{output_dir}} --propagation-horizon {{propagation_horizon}}

propagation-probe:
    uv run python scripts/probe_propagation.py --config configs/synthetic_input.yaml --warmup-steps 200 --horizon 512 --output-csv experiments/logs/propagation_probe.csv --output-summary experiments/logs/propagation_probe_summary.json

propagation-probe-responsive:
    uv run python scripts/probe_propagation.py --config configs/responsive_audio.yaml --warmup-steps 200 --horizon 512 --output-csv experiments/logs/responsive_propagation_probe.csv --output-summary experiments/logs/responsive_propagation_probe_summary.json

propagation-probe-custom config warmup horizon output_csv output_summary:
    uv run python scripts/probe_propagation.py --config {{config}} --warmup-steps {{warmup}} --horizon {{horizon}} --output-csv {{output_csv}} --output-summary {{output_summary}}

metrics-custom config steps sample_interval output:
    uv run python scripts/run_long_metrics.py --config {{config}} --steps {{steps}} --sample-interval {{sample_interval}} --output {{output}}

checkpoint:
    uv run python scripts/save_checkpoint.py --config configs/default.yaml --steps 10000 --output experiments/states/default-10k.npz

checkpoint-custom config steps output:
    uv run python scripts/save_checkpoint.py --config {{config}} --steps {{steps}} --output {{output}}

resume-checkpoint checkpoint steps output:
    uv run python scripts/resume_checkpoint.py --checkpoint {{checkpoint}} --steps {{steps}} --output {{output}}

benchmark:
    uv run python scripts/benchmark_field.py --sizes 64,128,200 --steps 1000 --repeats 3 --output outputs/benchmarks/field-step.csv

benchmark-custom sizes steps repeats output:
    uv run python scripts/benchmark_field.py --sizes {{sizes}} --steps {{steps}} --repeats {{repeats}} --output {{output}}

audio:
    uv run python scripts/render_audio_demo.py --config configs/default.yaml --duration-seconds 5 --output experiments/audio/default-demo.wav

audio-demo:
    uv run python scripts/render_audio_demo.py --config configs/audio_demo.yaml --duration-seconds 10 --output experiments/audio/audio-demo.wav

audio-bright:
    uv run python scripts/render_audio_demo.py --config configs/audio_demo.yaml --duration-seconds 10 --carrier-frequency 330 --frequency-scale 1.5 --output experiments/audio/audio-bright-demo.wav

audio-custom config duration output:
    uv run python scripts/render_audio_demo.py --config {{config}} --duration-seconds {{duration}} --output {{output}}

audio-custom-carrier config duration carrier frequency_scale output:
    uv run python scripts/render_audio_demo.py --config {{config}} --duration-seconds {{duration}} --carrier-frequency {{carrier}} --frequency-scale {{frequency_scale}} --output {{output}}

audio-steps steps:
    uv run python scripts/render_audio_demo.py --config configs/default.yaml --steps {{steps}} --output experiments/audio/default-{{steps}}-steps.wav

audio-live:
    uv run python scripts/play_audio_demo.py --config configs/audio_demo.yaml

audio-live-test:
    uv run python scripts/play_audio_demo.py --config configs/audio_demo.yaml --duration-seconds 5

audio-live-gated:
    uv run python scripts/play_audio_demo.py --config configs/audio_demo.yaml --audio-mode gated

audio-live-gated-test:
    uv run python scripts/play_audio_demo.py --config configs/audio_demo.yaml --audio-mode gated --duration-seconds 5

audio-live-event:
    uv run python scripts/play_audio_demo.py --config configs/audio_demo.yaml --audio-mode event

audio-live-event-test:
    uv run python scripts/play_audio_demo.py --config configs/audio_demo.yaml --audio-mode event --duration-seconds 10

audio-live-responsive:
    uv run python scripts/play_audio_demo.py --config configs/responsive_audio.yaml --audio-mode slope

audio-live-responsive-test:
    uv run python scripts/play_audio_demo.py --config configs/responsive_audio.yaml --audio-mode slope --duration-seconds 10

audio-live-custom config carrier frequency_scale gain:
    uv run python scripts/play_audio_demo.py --config {{config}} --carrier-frequency {{carrier}} --frequency-scale {{frequency_scale}} --gain {{gain}}

live:
    uv run python scripts/live_field.py --config configs/default.yaml --interval-ms 30 --steps-per-update 1

live-input:
    uv run python scripts/live_field.py --config configs/synthetic_input.yaml --interval-ms 30 --steps-per-update 2

live-audio:
    uv run python scripts/live_field.py --config configs/responsive_audio.yaml --interval-ms 30 --steps-per-update 2 --audio --audio-mode slope

live-audio-gated:
    uv run python scripts/live_field.py --config configs/synthetic_input.yaml --interval-ms 30 --steps-per-update 2 --audio --audio-mode gated

live-audio-synthetic:
    uv run python scripts/live_field.py --config configs/synthetic_input.yaml --interval-ms 30 --steps-per-update 2 --audio --audio-mode event

live-audio-event:
    uv run python scripts/live_field.py --config configs/responsive_audio.yaml --interval-ms 30 --steps-per-update 2 --audio --audio-mode event

live-audio-continuous:
    uv run python scripts/live_field.py --config configs/responsive_audio.yaml --interval-ms 30 --steps-per-update 2 --audio --audio-mode continuous

live-field:
    uv run python scripts/live_field.py --config configs/field_only.yaml --interval-ms 30 --steps-per-update 2

live-custom config interval_ms steps_per_update:
    uv run python scripts/live_field.py --config {{config}} --interval-ms {{interval_ms}} --steps-per-update {{steps_per_update}}

live-record config output:
    uv run python scripts/live_field.py --config {{config}} --interval-ms 30 --steps-per-update 2 --audio --audio-mode slope --diagnostics-output {{output}}

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
