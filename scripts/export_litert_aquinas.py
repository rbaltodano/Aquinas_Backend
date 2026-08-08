"""Export the fused Aquinas Gemma 4 checkpoint to a LiteRT-LM package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf.core.exportable_module_config import (
    ExportTask,
)


MINIMUM_FREE_BYTES = 40 * 1024**3
EXPECTED_ARCHITECTURE = "Gemma4ForConditionalGeneration"
SUPPORTED_QUANTIZATION_RECIPES = (
    "dynamic_wi4_afp32",
    "dynamic_wi8_emb4_afp32",
    "dynamic_wi8_afp32",
)


def validate_source(source: Path, output: Path) -> None:
    required_files = (
        source / "config.json",
        source / "tokenizer.json",
        source / "tokenizer_config.json",
        source / "chat_template.jinja",
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Source checkpoint is incomplete:\n{formatted}")
    if not (
        (source / "model.safetensors").is_file()
        or (source / "model.safetensors.index.json").is_file()
    ):
        raise SystemExit(
            "Source checkpoint has neither model.safetensors nor a sharded "
            "model index."
        )

    config = json.loads((source / "config.json").read_text())
    architectures = config.get("architectures", [])
    if EXPECTED_ARCHITECTURE not in architectures:
        raise SystemExit(
            f"Expected {EXPECTED_ARCHITECTURE}; found {architectures!r}."
        )
    if config.get("model_type") != "gemma4":
        raise SystemExit(
            f"Expected model_type 'gemma4'; found {config.get('model_type')!r}."
        )

    output.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < MINIMUM_FREE_BYTES:
        free_gib = free_bytes / 1024**3
        raise SystemExit(
            "LiteRT conversion needs at least 40 GiB of free working space; "
            f"only {free_gib:.1f} GiB is available."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/Aquinas-Final-HF"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/Aquinas-Final-LiteRT"),
    )
    parser.add_argument(
        "--quantization-recipe",
        choices=SUPPORTED_QUANTIZATION_RECIPES,
        default="dynamic_wi4_afp32",
        help=(
            "Decoder quantization. dynamic_wi8_emb4_afp32 is the higher-precision "
            "phone candidate: 8-bit fully connected weights with 4-bit embeddings."
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    validate_source(source, output)

    export.export(
        model=str(source),
        output_dir=str(output),
        task=ExportTask.IMAGE_TEXT_TO_TEXT,
        prefill_lengths=[128],
        cache_length=4_096,
        quantization_recipe=args.quantization_recipe,
        externalize_embedder=True,
        use_jinja_template=True,
        jinja_chat_template_override=str(source / "chat_template.jinja"),
        bundle_litert_lm=True,
        export_vision_encoder=True,
        vision_encoder_quantization_recipe="dynamic_wi8_afp32",
        experimental_lightweight_conversion=True,
    )


if __name__ == "__main__":
    main()
