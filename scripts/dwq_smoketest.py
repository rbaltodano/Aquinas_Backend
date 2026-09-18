"""Standalone DWQ smoke test for the Aquinas Gemma 4 checkpoint.

Bypasses mlx_lm.quant.dwq's CLI `load()` call (hardcoded strict=True) because
this checkpoint's KV-shared layers (config.num_kv_shared_layers=20) still
carry unused duplicate k_proj/v_proj/k_norm weights in the HF safetensors
that mlx_lm's Gemma4 module graph never instantiates for those layers.
"""

import copy
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optimizers

from mlx_lm.utils import load_model, load_tokenizer, quantize_model, save
from mlx_lm.quant.dwq import load_data, dwq_quantize

MODEL_PATH = Path("models/Aquinas-Final-HF")
OUT_PATH = "models/dwq_smoketest"
NUM_SAMPLES = 16
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 512

print("Loading model (strict=False to drop unused shared-KV weights)...", flush=True)
model, config = load_model(MODEL_PATH, lazy=True, strict=False)
tokenizer = load_tokenizer(MODEL_PATH)
print("Model loaded.", flush=True)

print("Loading data...", flush=True)
train_data, valid_data = load_data(
    tokenizer, "data/training_data", NUM_SAMPLES, MAX_SEQ_LENGTH
)
print(f"train={len(train_data)} valid={len(valid_data)}", flush=True)

print("Quantizing student model...", flush=True)
q_model = copy.deepcopy(model)
_, q_config = quantize_model(q_model, config, group_size=64, bits=4)

if mx.metal.is_available():
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(max_rec_size)

opt = optimizers.Adam(learning_rate=1e-6, bias_correction=True)


def target_fn(batch, idx, split):
    return model(batch)


print("Running DWQ correction loop...", flush=True)
dwq_quantize(
    q_model,
    target_fn,
    opt,
    train_data,
    valid_data,
    batch_size=BATCH_SIZE,
    max_seq_length=MAX_SEQ_LENGTH,
    seed=123,
    gradient_checkpoint=True,
)

print("Saving...", flush=True)
save(OUT_PATH, str(MODEL_PATH), q_model, tokenizer, q_config)
print("Done.", flush=True)
