from pathlib import Path

from neuroacoustic_resonator.cli.protocol import (
    record_protocol,
    replay_protocol,
)


def main() -> None:
    output = Path("outputs") / "protocol" / "example.jsonl"
    recorded = record_protocol(
        Path("configs") / "field_only.yaml",
        output,
        steps=16,
    )
    replayed = replay_protocol(output)
    if recorded.frames != replayed.frames:
        msg = "recorded and replayed protocol frames differ"
        raise RuntimeError(msg)
    print(f"Round trip preserved {len(replayed.frames)} frames: {output}")


if __name__ == "__main__":
    main()
