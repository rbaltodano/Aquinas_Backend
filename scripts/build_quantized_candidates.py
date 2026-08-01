"""Build non-destructive 6-bit and 4-bit copies of the Aquinas checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def build_candidate(source: Path, destination: Path, bits: int) -> None:
    if destination.exists():
        print(f"Skipping existing candidate: {destination}")
        return
    command = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        str(source),
        "--mlx-path",
        str(destination),
        "--quantize",
        "--q-bits",
        str(bits),
        "--q-group-size",
        "64",
        "--q-mode",
        "affine",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/Aquinas-Final"),
    )
    parser.add_argument(
        "--models-directory",
        type=Path,
        default=Path("models"),
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source checkpoint does not exist: {args.source}")

    build_candidate(
        args.source,
        args.models_directory / "Aquinas-Final-6bit",
        bits=6,
    )
    build_candidate(
        args.source,
        args.models_directory / "Aquinas-Final-4bit",
        bits=4,
    )


if __name__ == "__main__":
    main()
