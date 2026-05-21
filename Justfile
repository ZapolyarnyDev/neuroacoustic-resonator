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

audio-input input output:
    uv run python scripts/extract_audio_input.py --input {{input}} --output {{output}}

audio-input-custom input output frame_size hop_size drive_strength:
    uv run python scripts/extract_audio_input.py --input {{input}} --output {{output}} --frame-size {{frame_size}} --hop-size {{hop_size}} --drive-strength {{drive_strength}}

audio-input-run input:
    uv run python scripts/run_audio_input.py --config configs/field_only.yaml --input {{input}} --output-csv experiments/logs/audio_input_run.csv --output-summary experiments/logs/audio_input_run_summary.json

audio-input-run-custom config input output_csv output_summary frame_size hop_size drive_strength:
    uv run python scripts/run_audio_input.py --config {{config}} --input {{input}} --output-csv {{output_csv}} --output-summary {{output_summary}} --frame-size {{frame_size}} --hop-size {{hop_size}} --drive-strength {{drive_strength}}

audio-input-run-propagated input:
    uv run python scripts/run_audio_input.py --config configs/field_only.yaml --input {{input}} --output-csv experiments/logs/audio_input_propagated.csv --output-summary experiments/logs/audio_input_propagated_summary.json --input-assoc-gain 0.8 --input-output-gain 0.0

voice-probe input:
    uv run python scripts/probe_voice_response.py --config configs/field_only.yaml --input {{input}} --output-dir experiments/logs --prefix voice_vs_silence

voice-probe-custom config input output_dir prefix frame_size hop_size drive_strength:
    uv run python scripts/probe_voice_response.py --config {{config}} --input {{input}} --output-dir {{output_dir}} --prefix {{prefix}} --frame-size {{frame_size}} --hop-size {{hop_size}} --drive-strength {{drive_strength}}

voice-probe-propagated input:
    uv run python scripts/probe_voice_response.py --config configs/field_only.yaml --input {{input}} --output-dir experiments/logs --prefix voice_vs_silence_propagated --input-assoc-gain 0.8 --input-output-gain 0.0

voice-memory-probe input:
    uv run python scripts/probe_voice_memory.py --config configs/field_only.yaml --input {{input}} --output-csv experiments/logs/voice_memory_probe.csv --output-summary experiments/logs/voice_memory_probe_summary.json --input-assoc-gain 0.8 --input-output-gain 0.0

voice-memory-probe-custom config input output_csv output_summary frame_size hop_size drive_strength pause_steps max_steps:
    uv run python scripts/probe_voice_memory.py --config {{config}} --input {{input}} --output-csv {{output_csv}} --output-summary {{output_summary}} --frame-size {{frame_size}} --hop-size {{hop_size}} --drive-strength {{drive_strength}} --pause-steps {{pause_steps}} --max-steps {{max_steps}}

conversation input:
    uv run python scripts/run_conversation.py --config configs/field_only.yaml --inputs {{input}} --output experiments/audio/voice-conversation.wav --summary experiments/logs/voice_conversation_summary.json --input-assoc-gain 0.8 --input-output-gain 0.0

conversation-custom config output summary +inputs:
    uv run python scripts/run_conversation.py --config {{config}} --inputs {{inputs}} --output {{output}} --summary {{summary}}

turn-detect input:
    uv run python scripts/detect_turns.py --input {{input}} --output-dir experiments/audio/turns

turn-detect-custom input output_dir threshold_ratio min_silence_ms:
    uv run python scripts/detect_turns.py --input {{input}} --output-dir {{output_dir}} --threshold-ratio {{threshold_ratio}} --min-silence-ms {{min_silence_ms}}

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

live-wav input:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2

live-wav-audio input:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2 --audio --audio-mode slope

live-wav-coupled input:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2 --audio --audio-mode coupled

live-wav-coupled-propagated input:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2 --audio --audio-mode coupled --input-assoc-gain 0.8 --input-output-gain 0.0

live-wav-voice-response input:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2 --audio --audio-mode voice-response --input-assoc-gain 0.8 --input-output-gain 0.0 --audio-gain 0.35

live-wav-record input output:
    uv run python scripts/live_field.py --config configs/field_only.yaml --input-wav {{input}} --interval-ms 30 --steps-per-update 2 --audio --audio-mode coupled --diagnostics-output {{output}}

diagnostics-export input:
    uv run python scripts/export_diagnostics.py --input {{input}}

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
