"""Semantic relatedness for the Insight Tree.

MiniLM owns numeric semantic comparison. The Aquinas language model owns
user-facing generation such as definitions, labels, and blended insights.
"""

from __future__ import annotations

from threading import Lock
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
Embedding = NDArray[np.float32]


class MiniLMRelatednessProvider:
    """Loads MiniLM once and provides normalized semantic comparisons."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimensions(self) -> int:
        model = self._require_model()
        dimensions = model.get_embedding_dimension()
        if dimensions is None:
            raise RuntimeError("MiniLM did not report its embedding dimensions.")
        return dimensions

    def load(self) -> None:
        """Load the configured model once. Later calls are no-ops."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is None:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )

    def embed(self, text: str) -> Embedding:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> Embedding:
        """Embed non-empty, short semantic descriptions as unit vectors."""
        cleaned = [self._clean_text(text) for text in texts]
        if not cleaned:
            raise ValueError("At least one text value is required.")

        model = self._require_model()
        with self._inference_lock:
            vectors = model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def compare(self, left: str, right: str) -> dict[str, float]:
        """Return raw cosine similarity plus tree-friendly derived values."""
        vectors = self.embed_many([left, right])
        return self.compare_embeddings(vectors[0], vectors[1])

    def compare_embeddings(
        self,
        left: Embedding,
        right: Embedding,
    ) -> dict[str, float]:
        """Compare two normalized embeddings without re-encoding their text."""
        if left.shape != right.shape:
            raise ValueError("Embeddings must have matching dimensions.")
        if left.shape != (self.dimensions,):
            raise ValueError(
                f"Expected {self.dimensions}-value embeddings, received {left.shape}."
            )

        similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
        return {
            "similarity": similarity,
            "relatedness": float(np.clip(similarity, 0.0, 1.0)),
            "distance": 1.0 - similarity,
        }

    def centroid(self, embeddings: Sequence[Embedding]) -> Embedding:
        """Return a normalized Node centroid from its member Insight vectors."""
        if not embeddings:
            raise ValueError("At least one embedding is required for a centroid.")

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional collection.")
        if matrix.shape[1] != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions}-value embeddings, received {matrix.shape[1]}."
            )

        center = matrix.mean(axis=0)
        magnitude = float(np.linalg.norm(center))
        if magnitude == 0:
            raise ValueError("Cannot normalize a zero-length centroid.")
        return np.asarray(center / magnitude, dtype=np.float32)

    def _require_model(self) -> SentenceTransformer:
        self.load()
        if self._model is None:
            raise RuntimeError("MiniLM failed to load.")
        return self._model

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Text cannot be empty.")
        return cleaned


relatedness_provider = MiniLMRelatednessProvider()
