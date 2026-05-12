from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
RegionName = Literal["input", "output", "assoc"]


@dataclass(frozen=True)
class RegionMasks:
    input: BoolArray
    output: BoolArray
    assoc: BoolArray

    def __post_init__(self) -> None:
        masks = {
            "input": self.input,
            "output": self.output,
            "assoc": self.assoc,
        }
        shapes = {mask.shape for mask in masks.values()}
        if len(shapes) != 1:
            msg = "region masks must have the same shape"
            raise ValueError(msg)

        shape = next(iter(shapes))
        if len(shape) != 2:
            msg = "region masks must be 2D"
            raise ValueError(msg)

        for name, mask in masks.items():
            if mask.dtype != np.bool_:
                msg = f"{name} mask must be boolean"
                raise ValueError(msg)
            if not np.any(mask):
                msg = f"{name} mask must not be empty"
                raise ValueError(msg)

        overlap = (
            self.input.astype(np.uint8)
            + self.output.astype(np.uint8)
            + self.assoc.astype(np.uint8)
        )
        if np.any(overlap > 1):
            msg = "region masks must not overlap"
            raise ValueError(msg)

    @classmethod
    def from_size(cls, size: int, *, edge_fraction: float = 0.2) -> RegionMasks:
        if size <= 2:
            msg = "size must be greater than 2"
            raise ValueError(msg)
        if not 0.0 < edge_fraction < 0.5:
            msg = "edge_fraction must be between 0 and 0.5"
            raise ValueError(msg)

        edge_width = max(1, int(round(size * edge_fraction)))
        if size - 2 * edge_width < 1:
            msg = "edge_fraction leaves no association region"
            raise ValueError(msg)

        shape = (size, size)
        input_mask = np.zeros(shape, dtype=np.bool_)
        output_mask = np.zeros(shape, dtype=np.bool_)
        assoc_mask = np.zeros(shape, dtype=np.bool_)

        input_mask[:, :edge_width] = True
        output_mask[:, -edge_width:] = True
        assoc_mask[:, edge_width : size - edge_width] = True

        return cls(input=input_mask, output=output_mask, assoc=assoc_mask)

    @property
    def shape(self) -> tuple[int, int]:
        return self.input.shape

    def mask(self, name: RegionName) -> BoolArray:
        if name == "input":
            return self.input.copy()
        if name == "output":
            return self.output.copy()
        return self.assoc.copy()

    def labels(self) -> NDArray[np.uint8]:
        labels = np.zeros(self.shape, dtype=np.uint8)
        labels[self.input] = 1
        labels[self.output] = 2
        labels[self.assoc] = 3
        return labels
