# Aquinas_Backend

FastAPI service exposing the local MLX Aquinas model, MiniLM semantic relatedness, and
conversation-scoped SQLite persistence to the iOS app.

## Read first

Before changing generation, embeddings, API contracts, or Insight Tree behavior, read
[`MODEL-INTEGRATION.md`](../Aquinas-Foundations/MODEL-INTEGRATION.md). It is the cross-repository
source of truth. Read [`INSIGHT-TREE.md`](../Aquinas-Foundations/INSIGHT-TREE.md) for tree semantics.

## Current checkpoint

- The live runtime loads `google/gemma-4-E2B-it` through MLX-VLM and applies
  `models/aquinas_adapters` to its language tower. `models/Aquinas-Final` remains the fused
  text-generation checkpoint used by evaluation/quantization workflows. Gemma 4 E2B has a
  131,072-token context window, and the full base runtime retains its image-understanding tower.
- Conversation output is structured JSON: a visible response, a safe user-facing approach
  summary, and validated key-term metadata. Automatic routing uses direct-JSON generation for
  routine questions and retains hidden deep reasoning for complex questions. The iOS client
  always requests the public approach summary; it is not raw chain-of-thought.
- Streaming is filtered. Raw model output and hidden scratch-work channels never go to clients.
- Conversation messages accept bounded image attachments. The backend validates and decodes them,
  then supplies at most the eight most recent images to Gemma 4 in transcript order.
- Contextual definitions can be cached by conversation, normalized term, and source-response hash.
- `/conversation/compact` creates a replacement checkpoint for older model context while the
  client keeps its visible transcript.
- `sentence-transformers/all-MiniLM-L6-v2` supplies normalized 384-dimensional embeddings.
- SQLite stores per-conversation definitions, Insights, Node Concepts, membership, embeddings,
  sparse Node edges, response-analysis results, provenance, and deletion tombstones.
- Automatic response analysis creates zero or one pivotal Node Concept, never an automatic
  Insight. Manually saved definitions are Insights.
- Question of the Day, Node labels, response-driven tree extraction, Midpoint candidate generation,
  and Make Node child generation are live structured operations.
- Question of the Day uses the preemptible direct-JSON background path. Midpoint has a separate
  3,200-token candidate budget plus a 1,200-token weighted-center pass for five or more sources.
  Producing and repairing five genuine shared-region candidates requires more room than lighter
  structured actions.
- Runtime prompt assembly in `main.py` is the sole authority for model identity. Tokenizer
  configuration contains no embedded persona or identity prompt, and legacy standalone prompt
  entry points have been removed.
- The fused Aquinas language weights have also been reconstructed as a canonical Hugging Face
  Gemma 4 checkpoint and exported as a multimodal LiteRT-LM package for iOS validation. The local
  artifact is `models/Aquinas-Final-LiteRT/model.litertlm`; it is 2,722,385,120 bytes with SHA-256
  `5cb26c8e29d52ecf3e2b651e590761fe593dddcab0cbcdac7dc0692605ee5569`. Models remain
  gitignored and must never be committed.

## Architecture

- [`server.py`](server.py) — FastAPI schemas, routes, lifespan startup, and HTTP error boundaries.
- [`main.py`](main.py) — loads the configured checkpoint, owns priority-aware serialized MLX
  generation, strips hidden thought output for non-streaming calls, and exposes streaming
  fragments only to the structured parser.
- [`generation_coordinator.py`](generation_coordinator.py) — gives foreground questions and
  definitions priority and cooperatively preempts background tree analysis.
- [`structured_generation.py`](structured_generation.py) — task prompts, JSON extraction,
  validation, filtered stream parsing, conversation personality injection, and one-repair-attempt
  flows for every structured operation.
- [`relatedness.py`](relatedness.py) — startup-loaded, locally cached MiniLM provider for
  normalized embeddings, cosine comparison, and centroids. Its inference is serialized.
- [`insight_tree.py`](insight_tree.py) — deterministic attach-or-create decision engine. The
  membership threshold is currently `0.40`.
- [`tree_store.py`](tree_store.py) — SQLite schema, atomic/idempotent response mutations,
  persistent assignment, alias promotion, centroid repair, sparse edges, budding, definition
  cache, and tombstones. Default database: `data/insight_tree.sqlite3`.
- [`tests`](tests) — focused structured-generation, relatedness, tree-engine, and persistent-store
  tests.
- [`evaluation/prompt_quality_cases.json`](evaluation/prompt_quality_cases.json) and
  [`scripts/evaluate_prompt_quality.py`](scripts/evaluate_prompt_quality.py) — end-to-end prompt
  contract catalog, objective checks, resumable real-model runner, and human-review report.
- `scripts/` — one-off conversion, training-data, fusion, benchmarking, and evaluation tools.
  Scripts must not write identity or chat-template overrides into tokenizer configuration.
- [`scripts/build_hf_aquinas_checkpoint.py`](scripts/build_hf_aquinas_checkpoint.py) maps the
  600 fused MLX language tensors into a canonical Hugging Face Gemma 4 checkpoint while retaining
  the base vision/audio assets. [`scripts/export_litert_aquinas.py`](scripts/export_litert_aquinas.py)
  performs the LiteRT-LM export; [`scripts/export_litert_aquinas_stage.py`](scripts/export_litert_aquinas_stage.py)
  resumes memory- and disk-heavy export stages in separate processes.
- `models/` — local weights. Never commit them.

Both MLX and MiniLM are process-scoped resources. Do not load either model per request. MLX
generation is intentionally serialized. Foreground work can signal background generation to yield
at a token boundary. The stream sends its `start` event only after generation acquires the model
and produces a fragment, allowing the client to distinguish queued work from active work.

## API contract

### Conversation and definitions

- `POST /conversation/respond` — validated complete structured response. Accepts 1–20
  user/assistant messages, optional `compacted_context`, `thinking_enabled`, and
  `generation_mode` (`automatic`, `fast`, or `deep`), plus the selected conversation
  `personality`. Messages may contain up to eight JPEG, PNG, or WebP image payloads. A direct
  definition request additionally requires an in-text `insight`.
- `POST /conversation/respond/stream` — NDJSON events: `start`, optional `thinking_summary`,
  `response_delta`, then validated `complete`; failures emit `error`.
- `POST /conversation/compact` — accepts the previous checkpoint plus up to 20 turns since it and
  returns `{ "summary": ... }`.
- `POST /concept/define` — uncached contextual definition.
- `POST /conversation/{conversation_id}/concept/lookup` — returns the matching cached definition
  or `null` without generation.
- `POST /conversation/{conversation_id}/concept/define` — returns the cached result if present or
  generates, validates, and saves it.
- `POST /home/question-of-the-day` — generates one grounded daily question from recent
  conversation context and at most four relevant Insights.
- `POST /ask` — legacy unstructured route; do not use for new iOS features.

### Relatedness and Insight Tree

- `GET /relatedness/health`
- `POST /relatedness/similarity` — 1–128 pairs.
- `POST /insight-tree/label-node`
- `POST /concept/blend` — returns five candidates; the client selects the candidate nearest the
  weighted embedding centroid.
- `POST /concept/children` — returns exactly three validated Make Node child Insights.
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
Importing `main.py` loads the full Gemma 4 E2B multimodal checkpoint plus the Aquinas adapter, so
tests that do not need the real model should continue injecting fake generators/providers and
avoid importing the live server.

Run the focused test suite with:

```sh
python -m unittest discover -s tests
```

Validate the complete prompt-quality catalog without loading the model:

```sh
python scripts/evaluate_prompt_quality.py --validate-only
```

Run the real end-to-end suite incrementally:

```sh
aquinas_env/bin/python scripts/evaluate_prompt_quality.py \
  --model-path models/Aquinas-Final \
  --output evaluation/results/prompt-quality-bf16.json \
  --resume
```

The JSON result retains full outputs and editable 1–5 human rubrics; a readable Markdown report is
written beside it. Objective checks assert contracts, not exact prose. Never use Aquinas to grade
its own philosophical/content quality.

The latency-quality workflow is intentionally non-destructive:

```sh
python scripts/build_quantized_candidates.py
python scripts/benchmark_latency_quality.py --model-path models/Aquinas-Final --output evaluation/results/bf16.json
python scripts/benchmark_latency_quality.py --model-path models/Aquinas-Final-6bit --output evaluation/results/6bit.json
python scripts/benchmark_latency_quality.py --model-path models/Aquinas-Final-4bit --output evaluation/results/4bit.json
python scripts/select_quantized_candidate.py --bf16 evaluation/results/bf16.json --six-bit evaluation/results/6bit.json --four-bit evaluation/results/4bit.json
```

Complete the blind-review fields in both candidate result files before selection. BF16 remains the
live default unless a fully reviewed candidate passes every quality and latency gate.

### LiteRT-LM iOS conversion

The iOS validation artifact uses a 4-bit dynamic-weight decoder, externalized embeddings, an 8-bit
vision encoder, a 4,096-token cache, a 128-token prefill signature, and an FP16 activation
preference for Metal delegation. Rebuild the canonical input checkpoint before exporting:

```sh
litert_conversion_env/bin/python scripts/build_hf_aquinas_checkpoint.py \
  --base /absolute/path/to/google/gemma-4-e2b-it/snapshot \
  --fused models/Aquinas-Final \
  --output models/Aquinas-Final-HF

litert_conversion_env/bin/python scripts/export_litert_aquinas.py \
  --source models/Aquinas-Final-HF \
  --output models/Aquinas-Final-LiteRT
```

The exporter needs substantial temporary disk space; it deliberately refuses to start below
40 GiB free. The staged helper can resume additional-model, vision, and packaging work in isolated
processes after the decoder/embedder artifacts exist. Local CPU inference validated the converted
weights with the response “Prudence is the virtue of acting wisely and avoiding sin.” A Mac GPU
result is not expected because this LiteRT toolchain has no WebGPU adapter on the development Mac;
the authoritative accelerated target is iPhone Metal.

## Safety and compatibility

- Pydantic schemas and structured parsers are client contracts. Update iOS decoding and
  `MODEL-INTEGRATION.md` in the same change when they move.
- Keep the one automatic JSON repair attempt for complete structured tasks.
- Repair only invalid contract surfaces, preserve valid substance, prefer `null` or omission for
  optional content, and fail after the single repair attempt.
- Structured application actions are neutral and personality-independent. Only ordinary
  conversation receives the selected personality instruction.
- Keep `main.py` as the only tokenizer chat-prompt assembler. The runtime rejects known legacy
  tokenizer-level identity overrides.
- Direct definition-response examples must display the required Insight object rather than
  contradicting the instruction with `null`. Never invent citations, quotations, source locations,
  or attributions in ordinary conversation.
- Never forward raw streamed checkpoint output to a client.
- Preserve stable response idempotency and atomic SQLite mutations.
- The permissive CORS and local HTTP configuration are development conveniences, not production
  deployment policy.
