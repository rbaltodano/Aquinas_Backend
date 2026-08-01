"""Reconstruct a canonical Hugging Face Gemma 4 checkpoint from MLX fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import struct
from typing import BinaryIO


COPY_CHUNK_BYTES = 16 * 1024 * 1024
MLX_LANGUAGE_PREFIX = "language_model.model."
HF_LANGUAGE_PREFIX = "model.language_model."


def read_safetensors_header(path: Path) -> tuple[int, dict[str, dict]]:
    with path.open("rb") as handle:
        header_length_bytes = handle.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header = json.loads(handle.read(header_length))
    return 8 + header_length, header


def tensor_entries(
    path: Path,
) -> tuple[int, dict[str, dict]]:
    data_start, header = read_safetensors_header(path)
    return data_start, {
        key: value
        for key, value in header.items()
        if key != "__metadata__"
    }


def mapped_hf_key(mlx_key: str) -> str:
    if not mlx_key.startswith(MLX_LANGUAGE_PREFIX):
        raise ValueError(f"Unexpected MLX tensor key: {mlx_key}")
    return mlx_key.replace(
        MLX_LANGUAGE_PREFIX,
        HF_LANGUAGE_PREFIX,
        1,
    )


def copy_tensor_bytes(
    source: BinaryIO,
    destination: BinaryIO,
    source_offset: int,
    destination_offset: int,
    byte_count: int,
) -> None:
    source.seek(source_offset)
    destination.seek(destination_offset)
    remaining = byte_count
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise EOFError("Source tensor ended before its declared byte range.")
        destination.write(chunk)
        remaining -= len(chunk)


def copy_support_files(base: Path, output: Path) -> None:
    for source in base.iterdir():
        if source.name == "model.safetensors":
            continue
        if source.is_file():
            shutil.copy2(source, output / source.name)


def build_checkpoint(base: Path, fused: Path, output: Path) -> None:
    base_model = base / "model.safetensors"
    fused_shards = sorted(fused.glob("model-*.safetensors"))
    if not base_model.is_file():
        raise SystemExit(f"Canonical base model is missing: {base_model}")
    if not fused_shards:
        raise SystemExit(f"No fused MLX shards found in {fused}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    destination_model = output / "model.safetensors"
    shutil.copy2(base_model, destination_model)
    copy_support_files(base, output)

    destination_start, destination_entries = tensor_entries(destination_model)
    fused_entries: dict[str, tuple[Path, int, dict]] = {}
    for shard in fused_shards:
        shard_start, entries = tensor_entries(shard)
        for mlx_key, metadata in entries.items():
            hf_key = mapped_hf_key(mlx_key)
            if hf_key in fused_entries:
                raise ValueError(f"Duplicate fused tensor: {hf_key}")
            fused_entries[hf_key] = (shard, shard_start, metadata)

    base_language_keys = {
        key
        for key in destination_entries
        if key.startswith(HF_LANGUAGE_PREFIX)
    }
    if set(fused_entries) != base_language_keys:
        missing = sorted(base_language_keys - set(fused_entries))
        unexpected = sorted(set(fused_entries) - base_language_keys)
        raise ValueError(
            "Language tensor mapping is incomplete. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    open_sources: dict[Path, BinaryIO] = {}
    try:
        with destination_model.open("r+b") as destination:
            for index, hf_key in enumerate(sorted(fused_entries), start=1):
                shard, source_start, source_metadata = fused_entries[hf_key]
                destination_metadata = destination_entries[hf_key]
                for field in ("dtype", "shape"):
                    if source_metadata[field] != destination_metadata[field]:
                        raise ValueError(
                            f"{hf_key} has mismatched {field}: "
                            f"{source_metadata[field]!r} vs "
                            f"{destination_metadata[field]!r}"
                        )

                source_begin, source_end = source_metadata["data_offsets"]
                destination_begin, destination_end = (
                    destination_metadata["data_offsets"]
                )
                source_size = source_end - source_begin
                destination_size = destination_end - destination_begin
                if source_size != destination_size:
                    raise ValueError(
                        f"{hf_key} has mismatched byte size: "
                        f"{source_size} vs {destination_size}"
                    )

                source = open_sources.get(shard)
                if source is None:
                    source = shard.open("rb")
                    open_sources[shard] = source
                copy_tensor_bytes(
                    source,
                    destination,
                    source_start + source_begin,
                    destination_start + destination_begin,
                    source_size,
                )
                if index % 50 == 0 or index == len(fused_entries):
                    print(
                        f"Patched {index}/{len(fused_entries)} language tensors."
                    )
    finally:
        for handle in open_sources.values():
            handle.close()

    manifest = {
        "base_checkpoint": str(base),
        "fused_checkpoint": str(fused),
        "language_tensor_count": len(fused_entries),
        "mapping": f"{MLX_LANGUAGE_PREFIX}* -> {HF_LANGUAGE_PREFIX}*",
    }
    (output / "aquinas_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--fused",
        type=Path,
        default=Path("models/Aquinas-Final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/Aquinas-Final-HF"),
    )
    args = parser.parse_args()
    build_checkpoint(
        args.base.resolve(),
        args.fused.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
