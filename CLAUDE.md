# Aquinas_Backend

FastAPI service exposing the local MLX Aquinas model, MiniLM semantic relatedness, and
conversation-scoped SQLite persistence to the iOS app.

## Read first

Before changing generation, embeddings, API contracts, or Insight Tree behavior, read
[`MODEL-INTEGRATION.md`](../Aquinas-Foundations/MODEL-INTEGRATION.md). It is the cross-repository
source of truth. Read [`INSIGHT-TREE.md`](../Aquinas-Foundations/INSIGHT-TREE.md) for tree semantics.

## Current checkpoint

- `models/Aquinas-Final` is the live fine-tuned MLX checkpoint.
- Conversation output is structured JSON: a visible response, a safe user-facing approach
  summary, and validated key-term metadata. The iOS client currently always requests the approach
  summary; it is not raw chain-of-thought.
- Streaming is filtered. Raw model output and hidden scratch-work channels never go to clients.
- Contextual definitions can be cached by conversation, normalized term, and source-response hash.
- `/conversation/compact` creates a replacement checkpoint for older model context while the
  client keeps its visible transcript.
- `sentence-transformers/all-MiniLM-L6-v2` supplies normalized 384-dimensional embeddings.
- SQLite stores per-conversation definitions, Insights, Node Concepts, membership, embeddings,
  sparse Node edges, response-analysis results, provenance, and deletion tombstones.
- Automatic response analysis creates zero or one pivotal Node Concept, never an automatic
  Insight. Manually saved definitions are Insights.
- Node labels and response-driven tree extraction are live. Midpoint blend and Make Node child
  generation routes are not implemented.

## Architecture

- [`server.py`](server.py) — FastAPI schemas, routes, lifespan startup, and HTTP error boundaries.
- [`main.py`](main.py) — loads `models/Aquinas-Final`, owns the serialized MLX generation lock,
  strips hidden thought output for non-streaming calls, and exposes streaming fragments only to
  the structured parser.
- [`structured_generation.py`](structured_generation.py) — task prompts, JSON extraction,
  validation, filtered stream parsing, and one-repair-attempt flows for conversations,
  compaction, definitions, Node labels, and response-driven tree extraction.
- [`relatedness.py`](relatedness.py) — startup-loaded, locally cached MiniLM provider for
  normalized embeddings, cosine comparison, and centroids. Its inference is serialized.
- [`insight_tree.py`](insight_tree.py) — deterministic attach-or-create decision engine. The
  membership threshold is currently `0.40`.
- [`tree_store.py`](tree_store.py) — SQLite schema, atomic/idempotent response mutations,
  persistent assignment, alias promotion, centroid repair, sparse edges, budding, definition
  cache, and tombstones. Default database: `data/insight_tree.sqlite3`.
- [`tests`](tests) — focused structured-generation, relatedness, tree-engine, and persistent-store
  tests.
- [`rag_chat.py`](rag_chat.py) and `aquinas_memory/` — older RAG/memory path; not the primary iOS
  conversation contract.
- `scripts/` — one-off conversion, training-data, fusion, and identity scripts.
- `models/` — local weights. Never commit them.

Both MLX and MiniLM are process-scoped resources. Do not load either model per request. MLX
generation is intentionally serialized; the stream sends its `start` event only after generation
actually acquires the model and produces a fragment, allowing the client to distinguish queued
work from active work.

## API contract

### Conversation and definitions

- `POST /conversation/respond` — validated complete structured response. Accepts 1–20
  user/assistant messages, optional `compacted_context`, and `thinking_enabled`.
- `POST /conversation/respond/stream` — NDJSON events: `start`, optional `thinking_summary`,
  `response_delta`, then validated `complete`; failures emit `error`.
- `POST /conversation/compact` — accepts the previous checkpoint plus up to 20 turns since it and
  returns `{ "summary": ... }`.
- `POST /concept/define` — uncached contextual definition.
- `POST /conversation/{conversation_id}/concept/lookup` — returns the matching cached definition
  or `null` without generation.
- `POST /conversation/{conversation_id}/concept/define` — returns the cached result if present or
  generates, validates, and saves it.
- `POST /ask` — legacy unstructured route; do not use for new iOS features.

### Relatedness and Insight Tree

- `GET /relatedness/health`
- `POST /relatedness/similarity` — 1–128 pairs.
- `POST /insight-tree/label-node`
- `POST /insight-tree/assign` — stateless diagnostic attach-or-create decision.
- `POST /insight-tree/{conversation_id}/insights` — persistent manual Insight assignment. It also
  promotes a semantically matching automatic alias when appropriate.
- `GET /insight-tree/{conversation_id}` — tree snapshot without raw embeddings.
- `POST /insight-tree/{conversation_id}/nodes/{node_id}/label`
- `POST /insight-tree/{conversation_id}/responses/{response_id}/analyze` — idempotent automatic
  Node Concept extraction/mutation keyed by stable response ID.
- `DELETE /insight-tree/{conversation_id}/insights/{insight_id}` — removes an Insight, repairs its
  former Node, and records a tombstone when needed.

Do not let Aquinas generate IDs or numeric relatedness. IDs come from application code; MiniLM
owns similarity; deterministic application code owns topology and persistence.

## Running

Use the project virtual environment:

```sh
source aquinas_env/bin/activate
pip install -r requirements.txt
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
uvicorn server:app --reload
```

MiniLM is downloaded once by the explicit command and then loaded with `local_files_only=True`.
Importing `main.py` loads the roughly 9 GB MLX checkpoint, so tests that do not need the real model
should continue injecting fake generators/providers and avoid importing the live server.

Run the focused test suite with:

```sh
python -m unittest discover -s tests
```

## Safety and compatibility

- Pydantic schemas and structured parsers are client contracts. Update iOS decoding and
  `MODEL-INTEGRATION.md` in the same change when they move.
- Keep the one automatic JSON repair attempt for complete structured tasks.
- Never forward raw streamed checkpoint output to a client.
- Preserve stable response idempotency and atomic SQLite mutations.
- The permissive CORS and local HTTP configuration are development conveniences, not production
  deployment policy.
