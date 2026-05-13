from __future__ import annotations

import json

from neuroacoustic_resonator.analysis.diagnostics_export import (
    export_diagnostics_artifacts,
    read_diagnostics_rows,
    summarize_diagnostics_rows,
)


def test_read_and_summarize_diagnostics_rows(tmp_path) -> None:
    csv_path = tmp_path / "diagnostics.csv"
    csv_path.write_text(
        "step,global_synchrony,mean_metabolite,output_activity,"
        "output_fast_activity,output_slow_activity,output_event_score,"
        "output_fast_response_score,output_slow_drift_score,input_value,"
        "audio_envelope\n"
        "1,0.1,0.9,0.2,0.3,0.4,0.01,0.02,0.03,0.5,0.6\n"
        "2,0.2,0.8,0.4,0.5,0.6,0.02,0.03,0.04,0.0,0.7\n",
        encoding="utf-8",
    )

    rows = read_diagnostics_rows(csv_path)
    summary = summarize_diagnostics_rows(rows)

    assert len(rows) == 2
    assert summary["rows"] == 2
    assert summary["last_step"] == 2
    assert summary["output_activity"]["max"] == 0.4


def test_export_diagnostics_artifacts_writes_png_and_summary(tmp_path) -> None:
    csv_path = tmp_path / "diagnostics.csv"
    plot_path = tmp_path / "diagnostics.png"
    summary_path = tmp_path / "diagnostics.json"
    csv_path.write_text(
        "step,global_synchrony,mean_metabolite,output_activity,"
        "output_fast_activity,output_slow_activity,output_event_score,"
        "output_fast_response_score,output_slow_drift_score,input_value,"
        "audio_envelope\n"
        "1,0.1,0.9,0.2,0.3,0.4,0.01,0.02,0.03,0.5,0.6\n"
        "2,0.2,0.8,0.4,0.5,0.6,0.02,0.03,0.04,0.0,0.7\n",
        encoding="utf-8",
    )

    summary = export_diagnostics_artifacts(
        csv_path,
        plot_path=plot_path,
        summary_path=summary_path,
    )
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["rows"] == 2
    assert plot_path.exists()
    assert loaded["outputs"]["plot"] == str(plot_path)
