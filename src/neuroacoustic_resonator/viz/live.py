from __future__ import annotations

import argparse
import csv
import importlib
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.analysis.metrics import (
    RegionalActivityMetrics,
    RegionalActivityTracker,
    compute_regional_activity_metrics,
)
from neuroacoustic_resonator.analysis.diagnostics_export import (
    export_diagnostics_artifacts,
)
from neuroacoustic_resonator.audio.input import (
    WavInputDrive,
    extract_audio_input_features,
)
from neuroacoustic_resonator.audio.output import (
    ContinuousAudioRenderer,
    EventDrivenAudioRenderer,
    GatedAudioRenderer,
    SlopeTriggeredAudioRenderer,
    StimulusCoupledAudioRenderer,
    VoiceResponseSonificationRenderer,
)
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.field import FieldState, FloatArray
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame


@dataclass(frozen=True)
class LiveVisualizationConfig:
    config_path: Path = Path("configs") / "default.yaml"
    interval_ms: int = 30
    steps_per_update: int = 1
    history_size: int = 512
    audio_enabled: bool = False
    audio_mode: str = "gated"
    audio_sample_rate: int = 48_000
    audio_frame_size: int = 512
    audio_gain: float = 0.2
    audio_carrier_frequency: float = 220.0
    audio_frequency_scale: float = 1.0
    audio_gate_threshold: float = 0.002
    audio_gate_sensitivity: float = 24.0
    audio_coupled_input_threshold: float = 0.08
    audio_coupled_input_onset_threshold: float = 0.025
    audio_coupled_retrigger_frames: int = 8
    audio_coupled_response_threshold: float = 0.0004
    audio_coupled_response_sensitivity: float = 260.0
    audio_coupled_response_window_frames: int = 14
    diagnostics_output_path: Path | None = None
    diagnostics_sample_interval: int = 1
    input_wav_path: Path | None = None
    input_frame_size: int = 1024
    input_hop_size: int = 512
    input_drive_strength: float = 0.45
    input_assoc_gain: float = 0.0
    input_output_gain: float = 0.0
    input_loop: bool = False

    def __post_init__(self) -> None:
        if self.interval_ms < 1:
            msg = "interval_ms must be positive"
            raise ValueError(msg)
        if self.steps_per_update < 1:
            msg = "steps_per_update must be positive"
            raise ValueError(msg)
        if self.history_size < 2:
            msg = "history_size must be at least 2"
            raise ValueError(msg)
        if self.audio_mode not in {
            "continuous",
            "gated",
            "event",
            "slope",
            "coupled",
            "voice-response",
        }:
            msg = (
                "audio_mode must be 'continuous', 'gated', 'event', "
                "'slope', 'coupled', or 'voice-response'"
            )
            raise ValueError(msg)
        if self.audio_sample_rate < 1:
            msg = "audio_sample_rate must be positive"
            raise ValueError(msg)
        if self.audio_frame_size < 1:
            msg = "audio_frame_size must be positive"
            raise ValueError(msg)
        if self.audio_gain < 0.0:
            msg = "audio_gain must be non-negative"
            raise ValueError(msg)
        if self.audio_carrier_frequency <= 0.0:
            msg = "audio_carrier_frequency must be positive"
            raise ValueError(msg)
        if self.audio_frequency_scale <= 0.0:
            msg = "audio_frequency_scale must be positive"
            raise ValueError(msg)
        if self.audio_gate_threshold < 0.0:
            msg = "audio_gate_threshold must be non-negative"
            raise ValueError(msg)
        if self.audio_gate_sensitivity <= 0.0:
            msg = "audio_gate_sensitivity must be positive"
            raise ValueError(msg)
        if self.audio_coupled_input_threshold < 0.0:
            msg = "audio_coupled_input_threshold must be non-negative"
            raise ValueError(msg)
        if self.audio_coupled_input_onset_threshold < 0.0:
            msg = "audio_coupled_input_onset_threshold must be non-negative"
            raise ValueError(msg)
        if self.audio_coupled_retrigger_frames < 0:
            msg = "audio_coupled_retrigger_frames must be non-negative"
            raise ValueError(msg)
        if self.audio_coupled_response_threshold < 0.0:
            msg = "audio_coupled_response_threshold must be non-negative"
            raise ValueError(msg)
        if self.audio_coupled_response_sensitivity <= 0.0:
            msg = "audio_coupled_response_sensitivity must be positive"
            raise ValueError(msg)
        if self.audio_coupled_response_window_frames < 1:
            msg = "audio_coupled_response_window_frames must be positive"
            raise ValueError(msg)
        if self.diagnostics_sample_interval < 1:
            msg = "diagnostics_sample_interval must be positive"
            raise ValueError(msg)
        if self.input_frame_size < 1:
            msg = "input_frame_size must be positive"
            raise ValueError(msg)
        if self.input_hop_size < 1:
            msg = "input_hop_size must be positive"
            raise ValueError(msg)
        if self.input_drive_strength < 0.0:
            msg = "input_drive_strength must be non-negative"
            raise ValueError(msg)
        if self.input_assoc_gain < 0.0:
            msg = "input_assoc_gain must be non-negative"
            raise ValueError(msg)
        if self.input_output_gain < 0.0:
            msg = "input_output_gain must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class VisualizationFrame:
    phase: FloatArray
    local_synchrony: FloatArray
    metabolite: FloatArray
    trace: FloatArray
    region_labels: np.ndarray
    step: int
    global_synchrony: float
    mean_metabolite: float
    regional_metrics: RegionalActivityMetrics


@dataclass(frozen=True)
class DiagnosticCurveSpec:
    key: str
    label: str
    color: tuple[int, int, int]
    group: str
    width: float = 1.5


DIAGNOSTIC_CSV_FIELDS = (
    "step",
    "global_synchrony",
    "mean_metabolite",
    "output_activity",
    "output_response_activity",
    "output_fast_activity",
    "output_slow_activity",
    "output_event_score",
    "output_fast_response_score",
    "output_slow_drift_score",
    "input_value",
    "audio_envelope",
    "stimulus_window",
    "coupled_audio_trigger",
)


def diagnostic_curve_specs() -> tuple[DiagnosticCurveSpec, ...]:
    return (
        DiagnosticCurveSpec(
            key="global_synchrony",
            label="global synchrony",
            color=(250, 220, 70),
            group="tonic",
        ),
        DiagnosticCurveSpec(
            key="mean_metabolite",
            label="mean metabolite",
            color=(70, 210, 230),
            group="tonic",
        ),
        DiagnosticCurveSpec(
            key="output_activity",
            label="output activity",
            color=(255, 135, 70),
            group="tonic",
            width=2.0,
        ),
        DiagnosticCurveSpec(
            key="output_response_activity",
            label="baseline response",
            color=(255, 165, 80),
            group="response",
            width=2.0,
        ),
        DiagnosticCurveSpec(
            key="output_event_score",
            label="output event score",
            color=(185, 135, 255),
            group="response",
        ),
        DiagnosticCurveSpec(
            key="output_fast_response_score",
            label="fast response score",
            color=(255, 210, 120),
            group="response",
        ),
        DiagnosticCurveSpec(
            key="output_slow_drift_score",
            label="slow drift score",
            color=(130, 170, 255),
            group="tonic",
        ),
        DiagnosticCurveSpec(
            key="input_value",
            label="input pulse |value|",
            color=(80, 235, 130),
            group="response",
        ),
        DiagnosticCurveSpec(
            key="audio_envelope",
            label="audio envelope",
            color=(245, 245, 245),
            group="response",
        ),
        DiagnosticCurveSpec(
            key="stimulus_window",
            label="stimulus window",
            color=(120, 255, 210),
            group="response",
        ),
        DiagnosticCurveSpec(
            key="coupled_audio_trigger",
            label="coupled trigger",
            color=(255, 120, 170),
            group="response",
        ),
    )


def diagnostic_curve_groups() -> tuple[
    tuple[str, str, tuple[DiagnosticCurveSpec, ...]], ...
]:
    specs = diagnostic_curve_specs()
    return (
        (
            "tonic",
            "tonic state",
            tuple(spec for spec in specs if spec.group == "tonic"),
        ),
        (
            "response",
            "response / events",
            tuple(spec for spec in specs if spec.group == "response"),
        ),
    )


def diagnostic_legend_html(specs: tuple[DiagnosticCurveSpec, ...]) -> str:
    rows = [
        "<div style='font-size: 11pt; font-weight: 600; color: #dddddd;'>diagnostics</div>"
    ]
    for group_key, group_label, group_specs in diagnostic_curve_groups():
        selected_specs = tuple(spec for spec in group_specs if spec in specs)
        if not selected_specs:
            continue
        rows.append(
            "<div style='font-size: 10pt; font-weight: 600; color: #9fd6ff; "
            "margin-top: 14px;'>"
            f"{group_label}"
            "</div>"
        )
        for spec in selected_specs:
            red, green, blue = spec.color
            rows.append(
                "<div style='font-size: 10pt; margin-top: 8px;'>"
                f"<span style='color: rgb({red}, {green}, {blue});'>--</span> "
                f"<span style='color: #d8d8d8;'>{spec.label}</span>"
                "</div>"
            )
        rows.append(f"<span style='display:none'>{group_key}</span>")
    return "".join(rows)


def region_boundary_columns(regions: RegionMasks) -> tuple[int, int]:
    labels = regions.labels()
    first_output_column = int(np.min(np.flatnonzero(np.any(regions.output, axis=0))))
    first_assoc_column = int(np.min(np.flatnonzero(np.any(regions.assoc, axis=0))))
    if not np.any(labels):
        msg = "regions must not be empty"
        raise ValueError(msg)
    return first_assoc_column, first_output_column


def frame_to_visualization(
    frame: SimulationFrame,
    regions: RegionMasks,
    *,
    input_value: float = 0.0,
    regional_metrics: RegionalActivityMetrics | None = None,
) -> VisualizationFrame:
    if frame.state.phase.shape != regions.shape:
        msg = "frame and regions must have matching shapes"
        raise ValueError(msg)

    return VisualizationFrame(
        phase=frame.state.phase,
        local_synchrony=frame.local_synchrony,
        metabolite=frame.state.metabolite,
        trace=frame.state.trace,
        region_labels=regions.labels(),
        step=frame.metrics.step,
        global_synchrony=frame.metrics.global_synchrony,
        mean_metabolite=frame.metrics.mean_metabolite,
        regional_metrics=regional_metrics
        or compute_regional_activity_metrics(
            frame,
            regions,
            input_value=input_value,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show a live pyqtgraph view of the oscillator field.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "default.yaml",
        help="YAML simulation config path.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=30,
        help="GUI update interval in milliseconds.",
    )
    parser.add_argument(
        "--steps-per-update",
        type=int,
        default=1,
        help="Simulation steps advanced per GUI update.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=512,
        help="Number of metric points retained in the live plots.",
    )
    parser.add_argument(
        "--audio", action="store_true", help="Enable live audio output."
    )
    parser.add_argument(
        "--audio-mode",
        choices=(
            "continuous",
            "gated",
            "event",
            "slope",
            "coupled",
            "voice-response",
        ),
        default="gated",
        help="Audio monitor mode used when --audio is enabled.",
    )
    parser.add_argument("--audio-sample-rate", type=int, default=48_000)
    parser.add_argument("--audio-frame-size", type=int, default=512)
    parser.add_argument("--audio-gain", type=float, default=0.2)
    parser.add_argument("--audio-carrier-frequency", type=float, default=220.0)
    parser.add_argument("--audio-frequency-scale", type=float, default=1.0)
    parser.add_argument("--audio-gate-threshold", type=float, default=0.002)
    parser.add_argument("--audio-gate-sensitivity", type=float, default=24.0)
    parser.add_argument("--audio-coupled-input-threshold", type=float, default=0.08)
    parser.add_argument(
        "--audio-coupled-input-onset-threshold",
        type=float,
        default=0.025,
    )
    parser.add_argument("--audio-coupled-retrigger-frames", type=int, default=8)
    parser.add_argument(
        "--audio-coupled-response-threshold", type=float, default=0.0004
    )
    parser.add_argument(
        "--audio-coupled-response-sensitivity",
        type=float,
        default=260.0,
    )
    parser.add_argument("--audio-coupled-response-window-frames", type=int, default=14)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Optional CSV path for live diagnostics snapshots.",
    )
    parser.add_argument(
        "--diagnostics-sample-interval",
        type=int,
        default=1,
        help="Write every Nth diagnostics row when --diagnostics-output is set.",
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        default=None,
        help="Optional offline WAV input routed into R_in instead of synthetic input.",
    )
    parser.add_argument("--input-frame-size", type=int, default=1024)
    parser.add_argument("--input-hop-size", type=int, default=512)
    parser.add_argument("--input-drive-strength", type=float, default=0.45)
    parser.add_argument(
        "--input-assoc-gain",
        type=float,
        default=0.0,
        help="Extra fraction of WAV input drive injected into R_assoc.",
    )
    parser.add_argument(
        "--input-output-gain",
        type=float,
        default=0.0,
        help="Extra fraction of WAV input drive injected into R_out.",
    )
    parser.add_argument(
        "--input-loop",
        action="store_true",
        help="Loop the extracted WAV drive after it reaches the end.",
    )
    return parser


def run_live_visualizer(config: LiveVisualizationConfig) -> int:
    from PySide6 import QtCore, QtWidgets

    pg = cast(Any, importlib.import_module("pyqtgraph"))

    simulation_config = SimulationConfig.from_file(config.config_path)
    simulation = Simulation.from_config(simulation_config)
    regions = RegionMasks.from_size(simulation_config.field.size)
    input_drive = build_live_input_drive(config, regions)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    visualizer = _LiveFieldWindow(
        app=app,
        qt_core=QtCore,
        pg=pg,
        simulation=simulation,
        regions=regions,
        input_drive=input_drive,
        config=config,
    )
    visualizer.show()
    exit_code = int(app.exec())
    if config.diagnostics_output_path is not None:
        try:
            export_diagnostics_artifacts(config.diagnostics_output_path)
        except ValueError:
            pass
    return exit_code


def build_live_input_drive(
    config: LiveVisualizationConfig,
    regions: RegionMasks,
) -> WavInputDrive | None:
    if config.input_wav_path is None:
        return None
    features = extract_audio_input_features(
        config.input_wav_path,
        frame_size=config.input_frame_size,
        hop_size=config.input_hop_size,
        drive_strength=config.input_drive_strength,
    )
    return WavInputDrive(
        features,
        regions,
        assoc_gain=config.input_assoc_gain,
        output_gain=config.input_output_gain,
    )


def step_simulation_with_wav_input(
    simulation: Simulation,
    input_drive: WavInputDrive,
    *,
    input_step: int,
) -> SimulationFrame:
    input_value = input_drive.apply(simulation.field, input_step)
    simulation.step_index += 1
    state = simulation.field.step()
    simulation.last_input_value = input_value
    return SimulationFrame(
        state=state,
        metrics=simulation.field.metrics(step=simulation.step_index),
        local_synchrony=simulation.field.local_synchrony(),
    )


class _LiveFieldWindow:
    def __init__(
        self,
        *,
        app: Any,
        qt_core: Any,
        pg: Any,
        simulation: Simulation,
        regions: RegionMasks,
        input_drive: WavInputDrive | None,
        config: LiveVisualizationConfig,
    ) -> None:
        self._app = app
        self._qt_core = qt_core
        self._pg = pg
        self._simulation = simulation
        self._regions = regions
        self._input_drive = input_drive
        self._config = config
        self._steps: deque[int] = deque(maxlen=config.history_size)
        self._diagnostic_specs = diagnostic_curve_specs()
        self._diagnostic_values: dict[str, deque[float]] = {
            spec.key: deque(maxlen=config.history_size)
            for spec in self._diagnostic_specs
        }
        self._diagnostic_curves: dict[str, Any] = {}
        self._diagnostic_plots: dict[str, Any] = {}
        self._diagnostics_legend_label: Any | None = None
        self._diagnostics_recorder = DiagnosticsSnapshotRecorder(
            config.diagnostics_output_path,
            sample_interval=config.diagnostics_sample_interval,
        )
        self._regional_tracker = RegionalActivityTracker()

        self._window = pg.GraphicsLayoutWidget(title="Neuroacoustic Resonator")
        self._window.setWindowTitle("Neuroacoustic Resonator - live field diagnostics")
        self._window.setBackground((5, 5, 5))
        self._window.resize(1680, 900)
        self._configure_layout_columns()
        self._phase_item = pg.ImageItem()
        self._synchrony_item = pg.ImageItem()
        self._metabolite_item = pg.ImageItem()
        self._trace_item = pg.ImageItem()
        self._regions_item = pg.ImageItem()
        self._boundary_columns = region_boundary_columns(regions)
        self._audio_output = (
            _LiveAudioOutput(config=config, regions=regions)
            if config.audio_enabled
            else None
        )

        for index, (title, item) in enumerate(
            (
                ("phase", self._phase_item),
                ("local synchrony", self._synchrony_item),
                ("metabolite", self._metabolite_item),
                ("trace", self._trace_item),
                ("regions", self._regions_item),
            )
        ):
            self._add_image_panel(title, item, row=index // 3, col=index % 3)

        self._add_diagnostics_plots()
        self._timer = qt_core.QTimer(self._window)
        self._timer.timeout.connect(self.update)

    def _configure_layout_columns(self) -> None:
        layout = self._window.ci.layout
        for column in range(3):
            layout.setColumnMinimumWidth(column, 360)
            layout.setColumnStretchFactor(column, 1)
        layout.setColumnMinimumWidth(3, 520)
        layout.setColumnStretchFactor(3, 2)
        layout.setColumnMinimumWidth(4, 190)
        layout.setColumnStretchFactor(4, 0)
        for row in range(2):
            layout.setRowMinimumHeight(row, 380)
            layout.setRowStretchFactor(row, 1)

    def _add_image_panel(self, title: str, item: Any, *, row: int, col: int) -> None:
        plot = self._window.addPlot(row=row, col=col, title=title)
        plot.setAspectLocked(True)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.addItem(item)
        for boundary in self._boundary_columns:
            line = self._pg.InfiniteLine(
                pos=boundary - 0.5,
                angle=90,
                pen=self._pg.mkPen((255, 80, 80), width=1),
            )
            plot.addItem(line)

    def _add_diagnostics_plots(self) -> None:
        for row, (group_key, title, specs) in enumerate(diagnostic_curve_groups()):
            plot = self._window.addPlot(row=row, col=3, title=title)
            self._configure_diagnostics_plot(plot)
            self._diagnostic_plots[group_key] = plot
            for spec in specs:
                curve = plot.plot(
                    pen=self._pg.mkPen(spec.color, width=spec.width),
                )
                self._diagnostic_curves[spec.key] = curve

        self._diagnostics_legend_label = self._pg.LabelItem(justify="left")
        self._diagnostics_legend_label.setText(
            diagnostic_legend_html(self._diagnostic_specs),
        )
        self._window.addItem(
            self._diagnostics_legend_label,
            row=0,
            col=4,
            rowspan=2,
        )

    def _configure_diagnostics_plot(self, plot: Any) -> None:
        plot.setMenuEnabled(False)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "step")
        plot.setLabel("left", "value")
        plot.getViewBox().setDefaultPadding(0.02)

    def show(self) -> None:
        self._window.show()
        if self._audio_output is not None:
            self._audio_output.start()
        self._qt_core.QTimer.singleShot(0, self.update)
        self._timer.start(self._config.interval_ms)

    def update(self) -> None:
        try:
            frame = self._simulation.snapshot()
            for _ in range(self._config.steps_per_update):
                frame = self._step_simulation()
            regional_metrics = self._regional_tracker.update(
                frame,
                self._regions,
                input_value=self._simulation.last_input_value,
            )
            view_frame = frame_to_visualization(
                frame,
                self._regions,
                regional_metrics=regional_metrics,
            )

            self._phase_item.setImage(view_frame.phase.T, autoLevels=True)
            self._synchrony_item.setImage(
                view_frame.local_synchrony.T,
                levels=(0.0, 1.0),
            )
            self._metabolite_item.setImage(view_frame.metabolite.T, levels=(0.0, 1.0))
            self._trace_item.setImage(view_frame.trace.T, autoLevels=True)
            self._regions_item.setImage(view_frame.region_labels.T, levels=(1, 3))

            self._steps.append(view_frame.step)
            if self._audio_output is not None:
                self._audio_output.update_state(
                    frame.state,
                    input_value=regional_metrics.input_value,
                    response_score=coupled_response_score(regional_metrics),
                )
            audio_envelope = (
                self._audio_output.envelope if self._audio_output is not None else 0.0
            )
            self._append_diagnostics(
                view_frame,
                audio_envelope,
                stimulus_window=(
                    self._audio_output.stimulus_window
                    if self._audio_output is not None
                    else 0.0
                ),
                coupled_audio_trigger=(
                    self._audio_output.coupled_audio_trigger
                    if self._audio_output is not None
                    else 0.0
                ),
            )
        except KeyboardInterrupt:
            self._timer.stop()
            if self._audio_output is not None:
                self._audio_output.stop()
            self._app.quit()
        except Exception:
            self._timer.stop()
            if self._audio_output is not None:
                self._audio_output.stop()
            traceback.print_exc()
            self._app.quit()

    def _step_simulation(self) -> SimulationFrame:
        if self._input_drive is None:
            return self._simulation.step()
        input_step = self._simulation.step_index
        if self._config.input_loop:
            input_step %= self._input_drive.features.frame_count
        return step_simulation_with_wav_input(
            self._simulation,
            self._input_drive,
            input_step=input_step,
        )

    def _append_diagnostics(
        self,
        view_frame: VisualizationFrame,
        audio_envelope: float,
        stimulus_window: float = 0.0,
        coupled_audio_trigger: float = 0.0,
    ) -> None:
        values = diagnostics_row(
            view_frame,
            audio_envelope,
            stimulus_window=stimulus_window,
            coupled_audio_trigger=coupled_audio_trigger,
        )
        step = int(values["step"])
        for key, value in values.items():
            if key == "step" or key not in self._diagnostic_curves:
                continue
            self._diagnostic_values[key].append(float(value))
            self._diagnostic_curves[key].setData(
                list(self._steps),
                list(self._diagnostic_values[key]),
            )
        self._diagnostics_recorder.record(values, step=step)


def diagnostics_row(
    view_frame: VisualizationFrame,
    audio_envelope: float,
    *,
    stimulus_window: float = 0.0,
    coupled_audio_trigger: float = 0.0,
) -> dict[str, float | int]:
    return {
        "step": view_frame.step,
        "global_synchrony": view_frame.global_synchrony,
        "mean_metabolite": view_frame.mean_metabolite,
        "output_activity": view_frame.regional_metrics.output_activity,
        "output_response_activity": (
            view_frame.regional_metrics.output_response_activity
        ),
        "output_fast_activity": view_frame.regional_metrics.output_fast_activity,
        "output_slow_activity": view_frame.regional_metrics.output_slow_activity,
        "output_event_score": view_frame.regional_metrics.output_event_score,
        "output_fast_response_score": (
            view_frame.regional_metrics.output_fast_response_score
        ),
        "output_slow_drift_score": view_frame.regional_metrics.output_slow_drift_score,
        "input_value": abs(view_frame.regional_metrics.input_value),
        "audio_envelope": audio_envelope,
        "stimulus_window": stimulus_window,
        "coupled_audio_trigger": coupled_audio_trigger,
    }


class DiagnosticsSnapshotRecorder:
    def __init__(
        self,
        output_path: Path | None,
        *,
        sample_interval: int = 1,
    ) -> None:
        if sample_interval < 1:
            msg = "sample_interval must be positive"
            raise ValueError(msg)
        self.output_path = output_path
        self.sample_interval = sample_interval
        self._initialized = False

    def record(self, row: dict[str, float | int], *, step: int) -> None:
        if self.output_path is None or step % self.sample_interval != 0:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._initialized or not self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=DIAGNOSTIC_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._initialized = True


class _LiveAudioOutput:
    def __init__(
        self,
        *,
        config: LiveVisualizationConfig,
        regions: RegionMasks,
        stream_factory: Any | None = None,
    ) -> None:
        self._config = config
        self._regions = regions
        self._lock = threading.Lock()
        self._state: FieldState | None = None
        self._input_value = 0.0
        self._response_score = 0.0
        self._stream: Any | None = None
        self._stream_factory = stream_factory
        self._renderer = self._build_renderer(config.audio_frame_size)
        self._last_status_text: str | None = None

    def _build_renderer(
        self, frame_size: int
    ) -> (
        ContinuousAudioRenderer
        | GatedAudioRenderer
        | EventDrivenAudioRenderer
        | SlopeTriggeredAudioRenderer
        | StimulusCoupledAudioRenderer
        | VoiceResponseSonificationRenderer
    ):
        if self._config.audio_mode == "gated":
            return GatedAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
                gate_threshold=self._config.audio_gate_threshold,
                gate_sensitivity=self._config.audio_gate_sensitivity,
            )
        if self._config.audio_mode == "event":
            return EventDrivenAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
            )
        if self._config.audio_mode == "slope":
            return SlopeTriggeredAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
            )
        if self._config.audio_mode == "coupled":
            return StimulusCoupledAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
                input_threshold=self._config.audio_coupled_input_threshold,
                input_onset_threshold=(
                    self._config.audio_coupled_input_onset_threshold
                ),
                retrigger_frames=self._config.audio_coupled_retrigger_frames,
                response_threshold=self._config.audio_coupled_response_threshold,
                response_sensitivity=self._config.audio_coupled_response_sensitivity,
                response_window_frames=(
                    self._config.audio_coupled_response_window_frames
                ),
            )
        if self._config.audio_mode == "voice-response":
            return VoiceResponseSonificationRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
            )
        return ContinuousAudioRenderer(
            sample_rate=self._config.audio_sample_rate,
            frame_size=frame_size,
            carrier_frequency=self._config.audio_carrier_frequency,
            frequency_scale=self._config.audio_frequency_scale,
            gain=self._config.audio_gain,
        )

    def start(self) -> None:
        if self._stream is not None:
            return
        factory = self._stream_factory
        if factory is None:
            sounddevice = cast(Any, importlib.import_module("sounddevice"))
            factory = sounddevice.OutputStream
        self._stream = factory(
            samplerate=self._config.audio_sample_rate,
            blocksize=self._config.audio_frame_size,
            channels=1,
            dtype="float32",
            callback=self.callback,
        )
        self._stream.__enter__()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.__exit__(None, None, None)
        self._stream = None

    def update_state(
        self,
        state: FieldState,
        *,
        input_value: float = 0.0,
        response_score: float = 0.0,
    ) -> None:
        with self._lock:
            self._state = state
            self._input_value = input_value
            self._response_score = response_score

    @property
    def envelope(self) -> float:
        return float(getattr(self._renderer, "envelope", 0.0))

    @property
    def stimulus_window(self) -> float:
        return float(getattr(self._renderer, "stimulus_window", 0.0))

    @property
    def coupled_audio_trigger(self) -> float:
        return float(getattr(self._renderer, "last_activation", 0.0))

    def callback(
        self,
        outdata: NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del time_info
        if status:
            self._report_status(status)

        with self._lock:
            state = self._state
            input_value = getattr(self, "_input_value", 0.0)
            response_score = getattr(self, "_response_score", 0.0)
        if state is None:
            outdata[:, 0] = 0.0
            return

        if frames != self._renderer.frame_size:
            self._renderer = self._build_renderer(frames)
        if isinstance(self._renderer, VoiceResponseSonificationRenderer):
            audio = self._renderer.render_frame(
                state,
                self._regions,
                response_score=response_score,
            )
        elif isinstance(self._renderer, StimulusCoupledAudioRenderer):
            audio = self._renderer.render_frame(
                state,
                self._regions,
                input_value=input_value,
                response_score=response_score,
            )
        else:
            audio = self._renderer.render_frame(state, self._regions)
        outdata[:, 0] = np.asarray(audio, dtype=np.float32)

    def _report_status(self, status: Any) -> None:
        status_text = str(status)
        if status_text == self._last_status_text:
            return
        self._last_status_text = status_text
        print(f"audio status: {status_text}")


def coupled_response_score(metrics: RegionalActivityMetrics) -> float:
    return max(
        metrics.output_fast_response_score,
        metrics.output_event_score,
        max(0.0, metrics.output_response_activity) * 0.05,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_live_visualizer(
        LiveVisualizationConfig(
            config_path=args.config,
            interval_ms=args.interval_ms,
            steps_per_update=args.steps_per_update,
            history_size=args.history_size,
            audio_enabled=args.audio,
            audio_mode=args.audio_mode,
            audio_sample_rate=args.audio_sample_rate,
            audio_frame_size=args.audio_frame_size,
            audio_gain=args.audio_gain,
            audio_carrier_frequency=args.audio_carrier_frequency,
            audio_frequency_scale=args.audio_frequency_scale,
            audio_gate_threshold=args.audio_gate_threshold,
            audio_gate_sensitivity=args.audio_gate_sensitivity,
            audio_coupled_input_threshold=args.audio_coupled_input_threshold,
            audio_coupled_input_onset_threshold=(
                args.audio_coupled_input_onset_threshold
            ),
            audio_coupled_retrigger_frames=args.audio_coupled_retrigger_frames,
            audio_coupled_response_threshold=args.audio_coupled_response_threshold,
            audio_coupled_response_sensitivity=args.audio_coupled_response_sensitivity,
            audio_coupled_response_window_frames=(
                args.audio_coupled_response_window_frames
            ),
            diagnostics_output_path=args.diagnostics_output,
            diagnostics_sample_interval=args.diagnostics_sample_interval,
            input_wav_path=args.input_wav,
            input_frame_size=args.input_frame_size,
            input_hop_size=args.input_hop_size,
            input_drive_strength=args.input_drive_strength,
            input_assoc_gain=args.input_assoc_gain,
            input_output_gain=args.input_output_gain,
            input_loop=args.input_loop,
        )
    )
