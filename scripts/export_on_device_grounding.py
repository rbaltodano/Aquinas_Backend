"""Exports the backend's Chroma grounding corpus into the flat format the iOS
on-device path expects (see Aquinas-iOS/Services/OnDeviceGroundingStore.swift
and MiniLMGroundingProvider.swift).

Produces two files, index-aligned with each other:
  - passages.json: an array of {text, title, sourceId, chunkIndex} objects
    (camelCase keys -- OnDeviceGroundingStore decodes with a plain
    JSONDecoder(), no snake_case conversion, so this must match its
    PassageRecord property names exactly).
  - embeddings.bin: raw float32, row-major, one 384-float row per passage in
    the same order as passages.json. OnDeviceGroundingStore validates
    embeddings.count == passages.count * 384 * 4 bytes at load time.

This is a one-time (re-run as the corpus grows) export, not a live sync --
the iOS app has no network dependency on the backend for this data.
"""

import json
import struct
from pathlib import Path

import chromadb

BACKEND_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = BACKEND_ROOT / "data" / "corpus" / "index"
COLLECTION_NAME = "aquinas_grounding"
OUTPUT_DIR = BACKEND_ROOT / "data" / "corpus" / "on_device_export"

EMBEDDING_DIMENSION = 384
BATCH_SIZE = 2000


def main():
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"Exporting {total} passages from '{COLLECTION_NAME}'...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    passages_path = OUTPUT_DIR / "passages.json"
    embeddings_path = OUTPUT_DIR / "embeddings.bin"

    passages: list[dict] = []
    with open(embeddings_path, "wb") as embeddings_file:
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            documents = batch["documents"]
            metadatas = batch["metadatas"]
            embeddings = batch["embeddings"]

            for document, metadata, embedding in zip(documents, metadatas, embeddings):
                if len(embedding) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSION}-dim embedding, got {len(embedding)} "
                        f"for source_id={metadata.get('source_id')}"
                    )
                passages.append({
                    "text": document,
                    "title": metadata.get("title") or metadata.get("source_id", "Unknown source"),
                    "sourceId": metadata.get("source_id", ""),
                    "chunkIndex": metadata.get("chunk_index", 0),
                })
                embeddings_file.write(struct.pack(f"<{EMBEDDING_DIMENSION}f", *embedding))

            offset += len(documents)
            print(f"  {offset}/{total}")

    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)

    expected_bytes = len(passages) * EMBEDDING_DIMENSION * 4
    actual_bytes = embeddings_path.stat().st_size
    print(f"Wrote {len(passages)} passages to {passages_path} ({passages_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote embeddings to {embeddings_path} ({actual_bytes / 1e6:.1f} MB)")
    assert actual_bytes == expected_bytes, (
        f"Byte count mismatch: expected {expected_bytes}, got {actual_bytes} "
        "-- OnDeviceGroundingStore's corpusMismatch check would reject this."
    )
    print("Byte-count check passed: embeddings.bin matches passages.json exactly.")


if __name__ == "__main__":
    main()
