set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    just --list

# Development
sync:
    uv sync --locked --dev

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

typecheck:
    uv run mypy src tests

test:
    uv run pytest

check: lint typecheck test

hooks-install:
    uv run pre-commit install

hooks:
    uv run pre-commit run --all-files

# Core field and research workflows
run:
    uv run python main.py

preview:
    uv run python main.py

metrics:
    uv run python scripts/run_long_metrics.py --config configs/default.yaml --steps 10000 --sample-interval 10 --output outputs/metrics/default-10k.csv

metrics-custom config steps sample_interval output:
    uv run python scripts/run_long_metrics.py --config {{config}} --steps {{steps}} --sample-interval {{sample_interval}} --output {{output}}

experiments:
    uv run python scripts/run_experiments.py --config configs/synthetic_input.yaml --output-dir experiments/logs --propagation-horizon 1024

propagation-probe:
    uv run python scripts/probe_propagation.py --config configs/synthetic_input.yaml --warmup-steps 200 --horizon 512 --output-csv experiments/logs/propagation_probe.csv --output-summary experiments/logs/propagation_probe_summary.json --output-plot experiments/logs/propagation_probe.png

pattern-calibration:
    uv run python scripts/run_pattern_calibration.py --config configs/field_only.yaml --synthetic tone:tone:220:0.5 --synthetic pulse:pulse:330:0.5 --synthetic chirp:chirp:180:0.5 --output-dir experiments/pattern_calibration --output-csv experiments/logs/pattern_calibration.csv --output-summary experiments/logs/pattern_calibration_summary.json --repeats 2 --input-assoc-gain 0.8 --input-output-gain 0.0

pattern-calibration-input input:
    uv run python scripts/run_pattern_calibration.py --config configs/field_only.yaml --input {{input}} --output-dir experiments/pattern_calibration --output-csv experiments/logs/pattern_calibration.csv --output-summary experiments/logs/pattern_calibration_summary.json --repeats 2 --input-assoc-gain 0.8 --input-output-gain 0.0

voice-probe input:
    uv run python scripts/probe_voice_response.py --config configs/field_only.yaml --input {{input}} --output-dir experiments/logs --prefix voice_vs_silence

voice-memory-probe input:
    uv run python scripts/probe_voice_memory.py --config configs/field_only.yaml --input {{input}} --output-csv experiments/logs/voice_memory_probe.csv --output-summary experiments/logs/voice_memory_probe_summary.json --input-assoc-gain 0.8 --input-output-gain 0.0

voice-memory-probe-control input:
    uv run python scripts/probe_voice_memory.py --config configs/field_only.yaml --input {{input}} --output-csv experiments/logs/voice_memory_probe_control.csv --output-summary experiments/logs/voice_memory_probe_control_summary.json --input-assoc-gain 0.8 --input-output-gain 0.0 --compare-silence-control

# User conversation workflows
audio-devices:
    uv run python scripts/run_live_conversation.py --list-devices

live-conversation preset="textured":
    uv run python scripts/run_live_conversation.py --config configs/field_only.yaml --preset {{preset}} --input-assoc-gain 0.8 --input-output-gain 0.0 --print-rms --record-dir experiments/audio/live-{{preset}}

live-conversation-levels:
    uv run python scripts/run_live_conversation.py --config configs/field_only.yaml --input-assoc-gain 0.8 --input-output-gain 0.0 --max-turns 1 --print-rms --idle-timeout-seconds 8

live-conversation-custom preset start_rms stop_rms:
    uv run python scripts/run_live_conversation.py --config configs/field_only.yaml --preset {{preset}} --start-rms {{start_rms}} --stop-rms {{stop_rms}} --input-assoc-gain 0.8 --input-output-gain 0.0 --print-rms --record-dir experiments/audio/live-{{preset}}

conversation input:
    uv run python scripts/run_conversation.py --config configs/field_only.yaml --inputs {{input}} --output experiments/audio/voice-conversation.wav --summary experiments/logs/voice_conversation_summary.json --input-assoc-gain 0.8 --input-output-gain 0.0

conversation-custom config output summary +inputs:
    uv run python scripts/run_conversation.py --config {{config}} --inputs {{inputs}} --output {{output}} --summary {{summary}}

turn-detect input:
    uv run python scripts/detect_turns.py --input {{input}} --output-dir experiments/audio/turns

# State and benchmark utilities
checkpoint:
    uv run python scripts/save_checkpoint.py --config configs/default.yaml --steps 10000 --output experiments/states/default-10k.npz

resume-checkpoint checkpoint steps output:
    uv run python scripts/resume_checkpoint.py --checkpoint {{checkpoint}} --steps {{steps}} --output {{output}}

benchmark:
    uv run python scripts/benchmark_field.py --sizes 64,128,200 --steps 1000 --repeats 3 --output outputs/benchmarks/field-step.csv

diagnostics-export input:
    uv run python scripts/export_diagnostics.py --input {{input}}

# Legacy audio demos: useful for regression checks, not the target user path.
legacy-audio-demo:
    uv run python scripts/render_audio_demo.py --config configs/audio_demo.yaml --duration-seconds 10 --output experiments/audio/audio-demo.wav

legacy-audio-live mode="slope":
    uv run python scripts/play_audio_demo.py --config configs/responsive_audio.yaml --audio-mode {{mode}}

legacy-live-field:
    uv run python scripts/live_field.py --config configs/field_only.yaml --interval-ms 30 --steps-per-update 2

legacy-live-wav input:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2
