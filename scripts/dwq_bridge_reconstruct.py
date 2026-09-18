"""Reconstruct a valid HF Gemma4ForConditionalGeneration checkpoint from a
DWQ-corrected MLX text-tower checkpoint, so it can go through the existing
litert_torch export pipeline (which loads via transformers.AutoModel and has
no notion of MLX's quantized tensor format).

DWQ (mlx_lm) only touches the language_model tower. This script:
  1. Dequantizes the DWQ-corrected language_model weights back to dense bf16.
  2. Renames mlx-lm's `language_model.model.X` keys to the checkpoint's real
     `model.language_model.X` naming.
  3. Splices in the ~60 shared-KV-layer self_attn tensors that mlx-lm's
     Gemma4 graph never materializes (has_kv=False for those layers) --
     copied verbatim from the original checkpoint since DWQ never touched
     them.
  4. Splices in vision_tower / audio_tower / embed_vision / embed_audio
     verbatim from the original checkpoint, since mlx-lm's Gemma4 support is
     text-only and never loaded them.

Output is a byte-for-byte-valid HF checkpoint directory with the DWQ
correction baked into dense weights, ready for the normal
export_litert_aquinas.py PTQ step to requantize.
"""

import shutil
from pathlib import Path

import mlx.core as mx

SOURCE_HF = Path("models/Aquinas-Final-HF")
DWQ_MODEL = Path("models/dwq_full_run_8bit")
OUTPUT = Path("models/dwq_bridged_hf_8bit")

DWQ_PREFIX = "language_model.model."
HF_LM_PREFIX = "model.language_model."


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    print("Reading DWQ safetensors...", flush=True)
    dwq_tensors = mx.load(str(DWQ_MODEL / "model.safetensors"))
    dwq_keys = list(dwq_tensors.keys())

    # Group quantized triples (weight/scales/biases) vs plain weights.
    quant_bases = set()
    plain_keys = []
    for k in dwq_keys:
        if k.endswith(".scales"):
            quant_bases.add(k[: -len(".scales")])
        elif k.endswith(".biases"):
            pass
        elif k.endswith(".weight"):
            pass
    for k in dwq_keys:
        if k.endswith(".weight") and k[: -len(".weight")] not in quant_bases:
            plain_keys.append(k)

    print(f"Quantized linear layers to dequantize: {len(quant_bases)}", flush=True)
    print(f"Plain (already dense) weights: {len(plain_keys)}", flush=True)

    reconstructed = {}

    for base in quant_bases:
        wq = dwq_tensors[base + ".weight"]
        scales = dwq_tensors[base + ".scales"]
        biases = dwq_tensors[base + ".biases"]
        dense = mx.dequantize(
            wq, scales=scales, biases=biases, group_size=64, bits=8
        )
        new_key = HF_LM_PREFIX + base[len(DWQ_PREFIX):] + ".weight"
        reconstructed[new_key] = dense.astype(mx.bfloat16)

    for k in plain_keys:
        new_key = HF_LM_PREFIX + k[len(DWQ_PREFIX):]
        reconstructed[new_key] = dwq_tensors[k].astype(mx.bfloat16)

    print(f"Reconstructed {len(reconstructed)} language_model tensors.", flush=True)

    print("Splicing in untouched tensors from the original checkpoint...", flush=True)
    orig_tensors = mx.load(str(SOURCE_HF / "model.safetensors"))
    orig_keys = set(orig_tensors.keys())
    missing = orig_keys - set(reconstructed.keys())
    print(f"Copying {len(missing)} verbatim tensors (shared-KV + vision/audio).", flush=True)
    for k in missing:
        reconstructed[k] = orig_tensors[k]

    assert set(reconstructed.keys()) == orig_keys, (
        f"Key mismatch: missing={orig_keys - set(reconstructed.keys())}, "
        f"extra={set(reconstructed.keys()) - orig_keys}"
    )
    print(f"Total reconstructed tensors: {len(reconstructed)} (matches original {len(orig_keys)})", flush=True)

    print("Writing safetensors (bf16)...", flush=True)
    mx.save_safetensors(str(OUTPUT / "model.safetensors"), reconstructed)

    print("Copying tokenizer/config/etc...", flush=True)
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "aquinas_conversion_manifest.json",
    ]:
        src = SOURCE_HF / name
        if src.exists():
            shutil.copy(src, OUTPUT / name)

    print("Done. Output at", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
