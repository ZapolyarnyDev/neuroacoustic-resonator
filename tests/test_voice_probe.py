from __future__ import annotations

import json

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.voice_probe import (
    VoiceVsSilenceProbeConfig,
    main,
    run_voice_vs_silence_probe,
)


def test_run_voice_vs_silence_probe_writes_comparison(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
  coupling_strength: 0.2
synthetic_input:
  enabled: false
steps: 8
""",
        encoding="utf-8",
    )
    wav_path = tmp_path / "voice.wav"
    samples = np.zeros(512, dtype=np.float32)
    samples[128:256] = 1.0
    wavfile.write(wav_path, 8_000, samples)

    summary = run_voice_vs_silence_probe(
        VoiceVsSilenceProbeConfig(
            config_path=config_path,
            input_wav=wav_path,
            output_dir=tmp_path,
            prefix="probe",
            frame_size=128,
            hop_size=64,
            warmup_steps=2,
            max_steps=5,
        )
    )

    loaded = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    assert summary["voice"]["peak_input_value"] > 0.0
    assert summary["silence"]["peak_input_value"] == 0.0
    assert "peak_fast_response" in summary["ratios"]
    assert loaded["voice"]["rows"] == 5


def test_voice_probe_main_writes_summary(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
synthetic_input:
  enabled: false
steps: 4
""",
        encoding="utf-8",
    )
    wav_path = tmp_path / "voice.wav"
    wavfile.write(wav_path, 8_000, np.ones(256, dtype=np.float32))

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--input",
            str(wav_path),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "probe",
            "--frame-size",
            "128",
            "--hop-size",
            "64",
            "--max-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "probe_summary.json").exists()
