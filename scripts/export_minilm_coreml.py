"""Converts sentence-transformers/all-MiniLM-L6-v2 to a Core ML model matching
the interface Aquinas-iOS/Services/MiniLMEmbedder.swift expects: inputs named
"input_ids" and "attention_mask" (int32, shape [1, 128]), a single output
that is already mean-pooled over real (non-padding) tokens and L2-normalized
-- so the Swift side does no pooling itself, just flattens the output array.

Run with the dedicated coreml_conversion_env (pinned to torch 2.7.0, the most
recent version coremltools 9.0 is actually tested against -- the main
aquinas_env's torch 2.13 fails to load coremltools' native prediction proxy
at all, which would make on-Mac verification impossible):

    Aquinas_Backend/coreml_conversion_env/bin/python3 scripts/export_minilm_coreml.py
"""

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEQUENCE_LENGTH = 128
BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_ROOT / "data" / "corpus" / "on_device_export"
OUTPUT_PATH = OUTPUT_DIR / "MiniLM.mlpackage"
VOCAB_OUTPUT_PATH = OUTPUT_DIR / "vocab.txt"


class MeanPoolNormalize(torch.nn.Module):
    """Wraps the base transformer with the same mean-pooling + L2-normalize
    sentence-transformers applies, baked into the graph so the Core ML model
    outputs a ready-to-use embedding directly."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Passing position_ids explicitly bypasses HF's internal dynamic-shape
        # position_ids slicing (self.position_ids[:, :seq_length]), which traces
        # as a non-scalar op that coremltools' torch frontend can't convert
        # ("TypeError: only 0-dimensional arrays can be converted to Python
        # scalars" in the `_int` op). Fixed sequence length makes this safe.
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        token_embeddings = output.last_hidden_state
        # Broadcasting (no explicit .expand()) avoids feeding a runtime-derived
        # tensor shape into an expand op, which traces as a dynamic value and
        # trips the same coremltools `_int` conversion bug the position_ids
        # and attention-implementation fixes above addressed elsewhere.
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # "eager" attention avoids transformers' newer unified masking/sdpa utility
    # code (masking_utils.py, sdpa_attention.py), which computes attention-mask
    # shape adjustments as dynamic ops that trip the same coremltools `_int`
    # conversion bug the explicit position_ids fix above addressed for embeddings.
    base_model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    base_model.eval()
    wrapped = MeanPoolNormalize(base_model)
    wrapped.eval()

    tokenizer.save_vocabulary(str(OUTPUT_DIR))
    # save_vocabulary writes "vocab.txt" directly under OUTPUT_DIR already,
    # matching WordPieceTokenizer.swift's expected filename.

    example_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQUENCE_LENGTH), dtype=torch.int64)
    example_mask = torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int64)

    print("Tracing...")
    traced = torch.jit.trace(wrapped, (example_ids, example_mask))

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, SEQUENCE_LENGTH), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, SEQUENCE_LENGTH), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        # coremltools defaults mlprogram conversion to FP16, whose CPU execution
        # path for this model is numerically broken: it returns NaN under
        # ComputeUnit.CPU_ONLY on macOS, and on the iOS Simulator (where
        # MiniLMEmbedder forces .cpuOnly, because the FP16 MPSGraph path is
        # device-only) it returns finite but badly wrong vectors -- correct
        # Summa hits scored ~0.50 against the corpus instead of ~0.84, which
        # silently degraded every on-device retrieval measurement taken on the
        # Simulator. FP32 keeps the CPU path correct; MiniLM is only 22M
        # parameters, so the bundle cost is small.
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(str(OUTPUT_PATH))
    print(f"Saved {OUTPUT_PATH}")

    # --- Verification: reference PyTorch vs. the converted Core ML model ---
    print("\nVerifying against the reference PyTorch model...")
    test_sentences = [
        "Whether God exists.",
        "The nature of the Trinity in Christian theology.",
        "I will come to you again, and take you to myself.",
        "A completely unrelated sentence about baking bread.",
    ]

    worst_cosine = 1.0
    for sentence in test_sentences:
        encoded = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            reference = wrapped(encoded["input_ids"], encoded["attention_mask"]).numpy()[0]

        coreml_input = {
            "input_ids": encoded["input_ids"].numpy().astype(np.int32),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int32),
        }
        coreml_output = mlmodel.predict(coreml_input)
        converted = list(coreml_output.values())[0][0]

        cosine = float(np.dot(reference, converted) / (np.linalg.norm(reference) * np.linalg.norm(converted)))
        worst_cosine = min(worst_cosine, cosine)
        print(f"  cosine={cosine:.6f}  {sentence!r}")

    print(f"\nWorst-case cosine similarity: {worst_cosine:.6f}")
    if worst_cosine < 0.999:
        raise SystemExit("Conversion fidelity below the 0.999 bar -- do not bundle this model as-is.")

    # The check above exercises whatever compute path Core ML picks by default,
    # which is why an FP16 export whose CPU path returned NaN/garbage once passed
    # it and shipped. The Simulator runs CPU-only, so verify that path explicitly.
    print("\nVerifying the CPU-only compute path (the one the iOS Simulator uses)...")
    cpu_model = ct.models.MLModel(str(OUTPUT_PATH), compute_units=ct.ComputeUnit.CPU_ONLY)
    worst_cpu_cosine = 1.0
    for sentence in test_sentences:
        encoded = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            reference = wrapped(encoded["input_ids"], encoded["attention_mask"]).numpy()[0]
        cpu_output = cpu_model.predict({
            "input_ids": encoded["input_ids"].numpy().astype(np.int32),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int32),
        })
        converted = list(cpu_output.values())[0][0]
        if not np.all(np.isfinite(converted)):
            raise SystemExit("CPU-only path produced non-finite output -- do not bundle this model.")
        cosine = float(np.dot(reference, converted) / (np.linalg.norm(reference) * np.linalg.norm(converted)))
        worst_cpu_cosine = min(worst_cpu_cosine, cosine)
        print(f"  cpu cosine={cosine:.6f}  {sentence!r}")

    print(f"\nWorst-case CPU-only cosine similarity: {worst_cpu_cosine:.6f}")
    if worst_cpu_cosine < 0.999:
        raise SystemExit("CPU-only fidelity below the 0.999 bar -- do not bundle this model as-is.")
    print("PASSED: Core ML export matches the reference model closely enough to bundle.")


if __name__ == "__main__":
    main()
