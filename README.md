# Aquinas Backend

The local FastAPI and MLX service used by Aquinas during development. It provides model
generation, structured response validation, semantic retrieval, Insight Tree operations, and
conversation-scoped SQLite persistence.

## Main areas

| Path | Responsibility |
| --- | --- |
| `server.py` | FastAPI routes, schemas, startup, and HTTP boundaries |
| `main.py` | MLX model loading and serialized generation |
| `structured_generation.py` | Prompt contracts, JSON validation, and filtered streaming |
| `generation_coordinator.py` | Foreground priority and background-task preemption |
| `grounding_retrieval.py` | Corpus retrieval and ranked evidence |
| `relatedness.py` | MiniLM embeddings and similarity |
| `insight_tree.py` / `tree_store.py` | Tree decisions and SQLite persistence |
| `tests/` | Focused contract and regression tests |
| `evaluation/` | Prompt-quality and retrieval evaluation data |
| `scripts/` | Corpus, conversion, export, and benchmarking tools |

The Home dashboard also reads optional development-time sections from this service: Loose Thread,
Terms You Glossed Over, Today in History, and Your Quote. The tree-analysis route returns before
the quote-notability check finishes; that best-effort check runs in the background and never
changes a completed tree-analysis response.

Read [`CLAUDE.md`](CLAUDE.md) for the current model checkpoint, API contract, and safety rules.
Read [`../Aquinas-Foundations/MODEL-INTEGRATION.md`](../Aquinas-Foundations/MODEL-INTEGRATION.md)
before changing a client-facing contract or model behavior.

## Local setup

```sh
source aquinas_env/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload
```

Run focused tests and contract validation with:

```sh
python -m unittest discover -s tests
python scripts/evaluate_prompt_quality.py --validate-only
```

Large model weights, generated corpora, databases, and evaluation outputs are local artifacts and
are intentionally excluded from source control.
