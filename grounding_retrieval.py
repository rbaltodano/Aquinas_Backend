"""Retrieval over the ingested grounding corpus (see ingest_corpus.py).

MiniLM embeds the query with the same provider used for Insight Tree
relatedness; Chroma returns the nearest passage chunks. This module only
retrieves passages — it never generates prose or validates claims against
them. That stays the model's and the prompt's job, matching the project's
"MiniLM compares meaning, Aquinas writes and explains" split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import chromadb

from relatedness import MiniLMRelatednessProvider

BACKEND_ROOT = Path(__file__).resolve().parent
INDEX_DIR = BACKEND_ROOT / "data" / "corpus" / "index"
COLLECTION_NAME = "aquinas_grounding"

# Cosine distance on MiniLM embeddings is a ranking signal, not a calibrated
# probability (see MODEL-INTEGRATION.md). Manual spot checks put genuinely
# on-topic passages under ~0.95 and clearly unrelated ones at ~1.10+, so this
# is a coarse screen against obviously irrelevant results, not a precision
# relevance threshold.
DEFAULT_MAX_DISTANCE = 1.0


@dataclass(frozen=True)
class GroundingPassage:
    text: str
    title: str
    source_id: str
    distance: float


class GroundingRetriever:
    """Loads the persisted Chroma collection once and serves nearest-passage queries."""

    def __init__(self) -> None:
        self._collection: "chromadb.Collection | None" = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None

    def load(self) -> None:
        """Open the persisted collection once. Later calls are no-ops."""
        if self._collection is not None:
            return
        with self._load_lock:
            if self._collection is None:
                client = chromadb.PersistentClient(path=str(INDEX_DIR))
                self._collection = client.get_or_create_collection(COLLECTION_NAME)

    def retrieve(
        self,
        query: str,
        embedder: MiniLMRelatednessProvider,
        k: int = 4,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ) -> list[GroundingPassage]:
        cleaned = query.strip()
        if not cleaned or self._collection is None:
            return []

        query_embedding = embedder.embed(cleaned).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        passages: list[GroundingPassage] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if distance > max_distance:
                continue
            passages.append(
                GroundingPassage(
                    text=document,
                    title=metadata.get("title") or metadata.get("source_id", "Unknown source"),
                    source_id=metadata.get("source_id", ""),
                    distance=distance,
                )
            )
        return passages


grounding_retriever = GroundingRetriever()
