from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if os.environ.get("NEUROACOUSTIC_BOOTSTRAPPED") != "1":
        candidates = (
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / ".venv" / "bin" / "python",
        )
        current_python = Path(sys.executable).resolve()
        for candidate in candidates:
            if candidate.exists() and candidate.resolve() != current_python:
                os.environ["NEUROACOUSTIC_BOOTSTRAPPED"] = "1"
                os.execv(str(candidate), [str(candidate), *sys.argv])

    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _main() -> int:
    _bootstrap()
    from neuroacoustic_resonator.audio_render import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
