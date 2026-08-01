"""Run memory- and disk-bounded stages of the Aquinas LiteRT-LM export."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.core.exportable_module_config import (
    ExportableModuleConfig,
    ExportTask,
)


def export_config(source: Path, work_dir: Path) -> ExportableModuleConfig:
    return ExportableModuleConfig(
        model=str(source),
        output_dir=str(work_dir.parent),
        work_dir=str(work_dir),
        task=ExportTask.IMAGE_TEXT_TO_TEXT,
        prefill_lengths=[128],
        cache_length=4_096,
        quantization_recipe="dynamic_wi4_afp32",
        externalize_embedder=True,
        use_jinja_template=True,
        jinja_chat_template_override=str(source / "chat_template.jinja"),
        bundle_litert_lm=True,
        export_vision_encoder=True,
        vision_encoder_quantization_recipe="dynamic_wi8_afp32",
        experimental_lightweight_conversion=True,
    )


def load_source(config: ExportableModuleConfig):
    source = export_lib.load_model(
        config.model,
        config,
        trust_remote_code=config.trust_remote_code,
        auto_model_override=config.auto_model_override,
        task=config.task,
    )
    return source, export_lib.update_export_config(config, source)


def run_decoder(source, config) -> None:
    artifacts = export_lib.export_text_prefill_decode_model(
        source,
        config,
        export_lib.ExportedModelArtifacts(),
    )
    print(artifacts.prefill_decode_model_path)


def run_embedder(source, config) -> None:
    artifacts = export_lib.export_embedder_model(
        source,
        config,
        export_lib.ExportedModelArtifacts(),
    )
    print(artifacts.embedder_model_path)


def run_additional(source, config) -> None:
    artifacts = export_lib.export_additional_models(
        source,
        config,
        export_lib.ExportedModelArtifacts(),
    )
    print(artifacts.additional_model_paths)


def run_vision(source, config) -> None:
    artifacts = export_lib.export_vision_encoder_models(
        source,
        config,
        export_lib.ExportedModelArtifacts(),
    )
    print(
        {
            "vision_encoder": artifacts.vision_encoder_model_path,
            "vision_adapter": artifacts.vision_adapter_model_path,
            "end_of_vision": artifacts.eoi_model_path,
        }
    )


def run_package(source, config, work_dir: Path) -> None:
    decoder_path = work_dir / "model_quantized.tflite"
    if not decoder_path.is_file():
        decoder_path = work_dir / "prefill_decode_quantized.tflite"
    artifacts = export_lib.ExportedModelArtifacts(
        prefill_decode_model_path=str(decoder_path),
        embedder_model_path=str(work_dir / "embedder_quantized.tflite"),
        vision_encoder_model_path=str(
            work_dir / "vision_encoder_quantized.tflite"
        ),
        vision_adapter_model_path=str(
            work_dir / "vision_adapter_quantized.tflite"
        ),
        additional_model_paths={
            "per_layer_embedder": str(
                work_dir / "per_layer_embedder_quantized.tflite"
            )
        },
    )
    required_paths = [
        Path(artifacts.prefill_decode_model_path),
        Path(artifacts.embedder_model_path),
        Path(artifacts.vision_encoder_model_path),
        Path(artifacts.vision_adapter_model_path),
        Path(artifacts.additional_model_paths["per_layer_embedder"]),
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing staged artifacts: {missing}")

    artifacts = export_lib.export_tokenizer(source, config, artifacts)
    # Tracing Gemma 4 with FP16 cache tensors currently fails in the nightly
    # exporter, while the exported graph itself supports LiteRT's FP16 Metal
    # delegation. The stock mobile package marks that graph with "fp16".
    # Override only the package metadata so the runtime selects FLOAT16.
    builder_type = litert_lm_builder.litertlm_builder.LitertLmFileBuilder
    original_add_model = builder_type.add_tflite_model

    def add_tflite_model_with_fp16(
        builder,
        model_path,
        model_type,
        *args,
        **kwargs,
    ):
        if (
            model_type
            == litert_lm_builder.litertlm_builder.TfLiteModelType.PREFILL_DECODE
        ):
            kwargs["prefer_activation_type"] = "fp16"
        return original_add_model(
            builder,
            model_path,
            model_type,
            *args,
            **kwargs,
        )

    with mock.patch.object(
        builder_type,
        "add_tflite_model",
        add_tflite_model_with_fp16,
    ):
        packaged = litert_lm_builder.package_model(source, config, artifacts)
    print(packaged.litert_lm_model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("decoder", "embedder", "additional", "vision", "package"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/Aquinas-Final-HF"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("models/Aquinas-Final-LiteRT/staged"),
    )
    args = parser.parse_args()
    source_path = args.source.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    config = export_config(source_path, work_dir)
    source, config = load_source(config)
    if args.stage == "decoder":
        run_decoder(source, config)
    elif args.stage == "embedder":
        run_embedder(source, config)
    elif args.stage == "additional":
        run_additional(source, config)
    elif args.stage == "vision":
        run_vision(source, config)
    else:
        run_package(source, config, work_dir)


if __name__ == "__main__":
    main()
