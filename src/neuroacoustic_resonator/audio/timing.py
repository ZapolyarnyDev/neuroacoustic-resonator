from __future__ import annotations

import math


def steps_for_duration(
    duration_seconds: float,
    *,
    sample_rate: int,
    frame_size: int,
) -> int:
    if duration_seconds <= 0.0:
        msg = "duration_seconds must be positive"
        raise ValueError(msg)
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if frame_size < 1:
        msg = "frame_size must be positive"
        raise ValueError(msg)

    return max(1, math.ceil(duration_seconds * sample_rate / frame_size))
