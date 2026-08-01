from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

BoundaryMode = Literal["open", "periodic"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class GridTopology:
    shape: tuple[int, int]
    boundary_x: BoundaryMode = "periodic"
    boundary_y: BoundaryMode = "periodic"

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(size < 2 for size in self.shape):
            msg = "grid shape must contain two dimensions greater than one"
            raise ValueError(msg)
        if self.boundary_x not in {"open", "periodic"}:
            msg = f"unsupported x boundary: {self.boundary_x!r}"
            raise ValueError(msg)
        if self.boundary_y not in {"open", "periodic"}:
            msg = f"unsupported y boundary: {self.boundary_y!r}"
            raise ValueError(msg)

    @classmethod
    def from_size(
        cls,
        size: int,
        *,
        boundary_x: BoundaryMode = "periodic",
        boundary_y: BoundaryMode = "periodic",
    ) -> GridTopology:
        return cls(
            (size, size),
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )

    def neighbor_count(self) -> FloatArray:
        count = np.zeros(self.shape, dtype=np.float64)
        for _, valid in self._neighbors(np.ones(self.shape, dtype=np.float64)):
            count += valid
        return count

    def neighbor_sum(self, values: FloatArray) -> FloatArray:
        self._require_shape(values)
        total = np.zeros(self.shape, dtype=np.float64)
        for neighbor, _ in self._neighbors(values):
            total += neighbor
        return total

    def laplacian(self, values: FloatArray) -> FloatArray:
        self._require_shape(values)
        return self.neighbor_sum(values) - self.neighbor_count() * values

    def phase_coupling(self, phase: FloatArray) -> FloatArray:
        self._require_shape(phase)
        total = np.zeros(self.shape, dtype=np.float64)
        for neighbor, valid in self._neighbors(phase):
            total += np.sin(neighbor - phase) * valid
        return total / self.neighbor_count()

    def local_phase_order(self, phase: FloatArray) -> FloatArray:
        self._require_shape(phase)
        total = np.exp(1j * phase)
        for neighbor, valid in self._neighbors(phase):
            total += np.exp(1j * neighbor) * valid
        return np.abs(total / (self.neighbor_count() + 1.0))

    def _neighbors(
        self,
        values: FloatArray,
    ) -> tuple[tuple[FloatArray, BoolArray], ...]:
        self._require_shape(values)
        return (
            self._shift(values, axis=0, amount=1, boundary=self.boundary_y),
            self._shift(values, axis=0, amount=-1, boundary=self.boundary_y),
            self._shift(values, axis=1, amount=1, boundary=self.boundary_x),
            self._shift(values, axis=1, amount=-1, boundary=self.boundary_x),
        )

    def _shift(
        self,
        values: FloatArray,
        *,
        axis: int,
        amount: int,
        boundary: BoundaryMode,
    ) -> tuple[FloatArray, BoolArray]:
        shifted = np.roll(values, amount, axis=axis)
        valid = np.ones(self.shape, dtype=np.bool_)
        if boundary == "periodic":
            return shifted, valid
        index: list[slice | int] = [slice(None), slice(None)]
        index[axis] = 0 if amount > 0 else -1
        edge = tuple(index)
        shifted[edge] = 0.0
        valid[edge] = False
        return shifted, valid

    def _require_shape(self, values: FloatArray) -> None:
        if values.shape != self.shape:
            msg = f"values shape must be {self.shape}, got {values.shape}"
            raise ValueError(msg)
