from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DiagnosticsRows = list[dict[str, float]]
DiagnosticsSummary = dict[str, Any]

DEFAULT_PLOT_FIELDS = (
    "global_synchrony",
    "mean_metabolite",
    "output_activity",
    "output_response_activity",
    "output_fast_response_score",
    "output_slow_drift_score",
    "input_value",
    "audio_envelope",
)


def export_diagnostics_artifacts(
    csv_path: str | Path,
    *,
    plot_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> DiagnosticsSummary:
    path = Path(csv_path)
    rows = read_diagnostics_rows(path)
    plot_output = Path(plot_path) if plot_path is not None else path.with_suffix(".png")
    summary_output = (
        Path(summary_path) if summary_path is not None else path.with_suffix(".json")
    )
    summary = summarize_diagnostics_rows(rows)
    summary["outputs"] = {
        "csv": str(path),
        "plot": str(plot_output),
        "summary": str(summary_output),
    }
    plot_diagnostics_rows(rows, plot_output)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_diagnostics_rows(csv_path: str | Path) -> DiagnosticsRows:
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in reader
        ]


def summarize_diagnostics_rows(rows: DiagnosticsRows) -> DiagnosticsSummary:
    if not rows:
        msg = "diagnostics rows must not be empty"
        raise ValueError(msg)

    summary: DiagnosticsSummary = {
        "rows": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
    }
    for key in rows[0]:
        if key == "step":
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "last": float(values[-1]),
        }
    return summary


def plot_diagnostics_rows(
    rows: DiagnosticsRows,
    output_path: str | Path,
    *,
    fields: tuple[str, ...] = DEFAULT_PLOT_FIELDS,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(12, 7))
    for field in fields:
        if field not in rows[0]:
            continue
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        axis.plot(steps, values, label=field)
    axis.set_title("Live diagnostics")
    axis.set_xlabel("step")
    axis.set_ylabel("value")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export PNG and JSON summary from a live diagnostics CSV.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_diagnostics_artifacts(
        args.input,
        plot_path=args.plot,
        summary_path=args.summary,
    )
    print(
        "Exported diagnostics: "
        f"rows={summary['rows']} "
        f"plot={summary['outputs']['plot']} "
        f"summary={summary['outputs']['summary']}"
    )
    return 0
