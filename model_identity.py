"""Canonical identity for the language model used by the Aquinas backend."""

import os


MODEL_BASE_ID = "google/gemma-4-E2B-it"
MODEL_DISPLAY_NAME = "Gemma 4 E2B"
MODEL_PATH = os.environ.get("AQUINAS_MODEL_PATH", "models/Aquinas-Final")
MODEL_RUNTIME_PATH = os.environ.get("AQUINAS_RUNTIME_MODEL_PATH", MODEL_BASE_ID)
MODEL_ADAPTER_PATH = os.environ.get(
    "AQUINAS_ADAPTER_PATH",
    "models/aquinas_adapters",
)
MODEL_CONTEXT_WINDOW_TOKENS = 131_072
