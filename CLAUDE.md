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
- Conversation messages may include one structured `insight_quote`. Prompt assembly escapes its
  title and definition into an `<insight_quote>` block immediately before the associated user
  question, while routing and direct-definition detection continue to inspect the plain question.
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
- Home can also surface optional Loose Thread, Terms You Glossed Over, Today in History, and Your
  Quote cards. Your Quote uses a cheap eligibility screen and a best-effort model assessment queued
  after a successful tree-analysis response; it must never delay or invalidate that response.
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
- The isolated replacement candidate
  `models/Qwen3-4B-LiteRT/qwen3_4b_mixed_int4.litertlm` is 2,659,057,664 bytes with SHA-256
  `f0794bc77efeaaf4f7af815f04c483b19b8f2ae4a102cef1b7b760a25848a18e`. It is rejected diagnostic
  input only: simulator Metal could not allocate its 388,956,160-byte tensor, and the base iPhone
  17 was terminated during initialization before generation. Do not promote, bundle, or fine-tune
  this package as the current replacement.
- Broad factual reliability is not a prompt-patching task. The planned path is automated ingestion
  of approved licensed/versioned sources, MiniLM passage retrieval, evidence-bound generation,
  citations, claim validation, and explicit uncertainty or approved online lookup when retrieval
  is insufficient. Fine-tuning should remain behavior/voice-focused.

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
- `POST /home/loose-thread`, `POST /home/glossed-terms`, `POST /home/today-in-history`, and
  `POST /home/your-quote` — optional Home discovery cards; each returns `null` when no suitable
  content exists. Today in History accepts an optional date override for deterministic tests.
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

For a text-only package, use `--skip-vision`. This avoids exporting the vision tower and adapter,
which is useful while on-device multimodal input remains post-launch work. Do not describe a
text-only artifact as supporting image input; the development backend can still accept bounded
image attachments through its full Gemma runtime.

The current LiteRT toolchain has no standard 6-bit Gemma recipe. The non-destructive
higher-precision phone candidate uses 8-bit fully connected weights with 4-bit embedding tables:

```sh
litert_conversion_env/bin/python scripts/export_litert_aquinas.py \
  --source models/Aquinas-Final-HF \
  --output models/Aquinas-Final-LiteRT-8fc4emb \
  --quantization-recipe dynamic_wi8_emb4_afp32
```

Never overwrite the working 4-bit artifact. A higher-precision package replaces it only after
package-size, cold-load, sustained-memory, latency, and blind answer-quality checks pass on the
base supported phone.

The August 1, 2026 `8fc4emb` candidate is 3,862,121,696 bytes with SHA-256
`9a6345f1a6cd39283f957977c84d31cc63b8dd56f2b8fffeb784940f63365282`, under
`models/Aquinas-Final-LiteRT-8fc4emb/`.

**August 3, 2026 correction — the original GPU rejection was a Simulator-only artifact, not a
real-device limitation. The candidate is confirmed working on GPU on the base supported iPhone.**
Its original rejection was based solely on an iOS Simulator probe failure
(`Failed to initialize kernel`). The stock (non-fine-tuned) `gemma-4-E2B-it.litertlm` from
`litert-community` on Hugging Face — byte-identical (2,588,147,712 bytes) to the copy in
`models/LiteRT-Stock/` — showed the same Simulator-only failure pattern (a Metal
texture-binding-limit error), yet Google's own published benchmark shows that exact file running
at 56.5 tokens/sec GPU decode on a real iPhone 17 Pro. Bumping the vendored LiteRT-LM runtime from
v0.12.0 to v0.14.0 (`Vendor/LiteRTLM/Package.swift` in `Aquinas-iOS`) did not change the Simulator
failure, confirming it's environment-specific, not a package or architecture-support issue.

Both the stock package and this `8fc4emb` candidate were then tested directly on a physical base
iPhone 17 (via `xcrun devicectl device copy to` into the app's Documents container, then
`--litert-probe --litert-probe-auto --litert-model-document`) and **both passed cleanly on GPU**:

- Stock `gemma-4-E2B-it.litertlm`: 2.59 GB, 3.89 s cold load, 0.78 s generation.
- `8fc4emb` candidate: 3.86 GB, 9.3 s cold load, 2.44 s generation. Roughly 2x the current 4-bit
  package's cold-load and generation time (4.33 s / 1.27 s) — expected for the precision increase,
  and still well within usable range.

(An earlier same-night attempt appeared to hang with no log output; that was a `devicectl`
console-streaming artifact, not a real failure — confirmed by a clean rerun with the phone
unlocked and the app foregrounded. Don't trust silence in a `devicectl --console` stream as a
hang; check the actual on-device UI.)

**Still not yet validated before this can replace the shipping manifest** — the project's own
gate is package-size, cold-load, sustained-memory, latency, *and* blind answer-quality on the base
supported phone. So far only a single one-shot probe generation has been measured. Missing:
sustained-memory/thermal behavior across a real multi-turn conversation (not just one probe
generation), and a proper blind answer-quality comparison against the current 4-bit package on
several prompts, not just "What is prudence?". Do not change the iOS manifest to this artifact
until those remaining gates pass.

This also means the llama.cpp migration (`Aquinas-Foundations/LLAMA-CPP-MIGRATION-SCOPING.md`)
may not be necessary — its entire premise was a LiteRT-LM GPU quality ceiling that turned out to
be a Simulator artifact, not a real one. Re-promoting this already-existing LiteRT package after
finishing the remaining validation gates is a much smaller change than a runtime migration.

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
