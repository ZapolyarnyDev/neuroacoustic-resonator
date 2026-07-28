from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from neuroacoustic_resonator.protocol.model import SoundProtocolFrame


@runtime_checkable
class ProtocolConsumer(Protocol):
    def consume(self, frame: SoundProtocolFrame) -> None: ...


class ProtocolPipeline:
    def __init__(self, consumers: Iterable[ProtocolConsumer] = ()) -> None:
        self._consumers = tuple(consumers)

    @property
    def consumers(self) -> tuple[ProtocolConsumer, ...]:
        return self._consumers

    def consume(self, frame: SoundProtocolFrame) -> None:
        for consumer in self._consumers:
            consumer.consume(frame)

    def run(self, frames: Iterable[SoundProtocolFrame]) -> int:
        count = 0
        for frame in frames:
            self.consume(frame)
            count += 1
        return count


class CaptureConsumer:
    def __init__(self) -> None:
        self.frames: list[SoundProtocolFrame] = []

    def consume(self, frame: SoundProtocolFrame) -> None:
        self.frames.append(frame)

    def clear(self) -> None:
        self.frames.clear()
