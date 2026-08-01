# Aquinas evaluation workflows

The backend has two complementary evaluation paths:

- `benchmark_latency_quality.py` compares checkpoint latency and conversation-answer parity.
- `evaluate_prompt_quality.py` exercises every structured prompt contract end to end.

## Prompt quality suite

The catalog in `prompt_quality_cases.json` covers ordinary conversation, contextual definitions,
Question of the Day, Node labels, response-driven Tree extraction, Midpoint candidates, Make Node
children, and conversation compaction. It includes representative and adversarial cases plus one
forced malformed-output repair probe for each operation.

Objective checks enforce properties such as required fields, exact counts, one-to-five-word Node
labels, grounded evidence, absence of definition metadata, candidate uniqueness, and exactly one
repair attempt. They intentionally do not assert exact prose.

Subjective qualities—accuracy, philosophical depth, contextual fit, neutrality, restraint,
candidate diversity, and preservation during repair—use explicit 1–5 human-review rubrics in the
generated report. A score of 3 is acceptable, 4 is strong, and 5 is excellent. Mark
`critical_failure` for hallucinated evidence, identity/persona leakage into a neutral action,
material philosophical error, unsafe hidden-reasoning exposure, or a result that violates the
product action even when its JSON is valid.

Validate the catalog without loading either model:

```sh
python scripts/evaluate_prompt_quality.py --validate-only
```

Run one operation:

```sh
aquinas_env/bin/python scripts/evaluate_prompt_quality.py \
  --model-path models/Aquinas-Final \
  --operation definition \
  --output evaluation/results/prompt-quality-bf16.json
```

Run the complete suite and calculate Midpoint selections with MiniLM:

```sh
aquinas_env/bin/python scripts/evaluate_prompt_quality.py \
  --model-path models/Aquinas-Final \
  --output evaluation/results/prompt-quality-bf16.json
```

The Markdown report is written beside the JSON unless `--report` supplies another path. Each case
is saved immediately. Use `--resume` after interruption; it skips completed objective passes while
retrying errors and objective failures. Use `--case-id` for focused reruns, `--operation` multiple
times to select several families, `--rerun-passing` after a prompt change, `--limit` for smoke
runs, or `--skip-minilm` when only candidate generation is under review.

Human scores live in the versioned `prompt_quality_reviews.json` catalog and are merged into result
JSON by case and rubric ID. After reviewing saved outputs, rebuild the summary and Markdown without
loading either model:

```sh
python scripts/evaluate_prompt_quality.py \
  --render-only \
  --output evaluation/results/prompt-quality-bf16.json
```

The suite passes only when every generation completes, every objective check passes, every rubric
is scored at least 3, and no case is marked as a critical failure. Do not use the generating
Aquinas checkpoint to grade its own substantive quality.

## Case-authoring rules

- Every case needs a unique ID, supported operation, input object, at least one objective check,
  and at least one human rubric.
- Prefer properties and rubrics over expected prose.
- Add a regression case whenever a prompt defect is found in real use.
- Keep neutral-action cases free from inputs that themselves call for slang or persona language.
- A repair probe must supply malformed but grounded material; it should test contract repair, not
  invite the model to invent missing subject matter.
- Keep the catalog stable when comparing checkpoints. Record model path, prompt revision, and
  review notes with results.
