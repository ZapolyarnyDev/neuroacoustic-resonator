from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
from numpy.typing import NDArray

AudioArray = NDArray[np.float64]


def write_wav(path: str | Path, audio: AudioArray, *, sample_rate: int) -> Path:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.asarray(np.round(clipped * 32767.0), dtype=np.int16)
    with wave.open(str(output_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output_path
