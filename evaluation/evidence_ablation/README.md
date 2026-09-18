# Evidence ablation experiment

Exploratory comparison of the same shipped Gemma package under automatically
retrieved passages (curated layer OFF), manually verified whole corpus chunks,
and no passages. Fifteen answerable questions have all three conditions;
current-pope and Vatican II questions have only retrieval/no-evidence conditions
because this historical corpus supplies no verified answer. Total: 49 jobs.

The production conversation system prompt is obtained through a Debug-only
accessor. Every instruction remains identical; only the evidence block changes.
Every request has empty conversation history. The runtime's deterministic
sampler, repetition guards and recovery behavior remain active. This bypasses
canned-answer shortcuts, definition routing, and the adapter's subsequent
accuracy audit: it measures evidence use through the runtime, not end-to-end app
answer quality. Conditions rotate order within questions. Manually selected
source chunks are test fixtures only; neither corpus nor production retrieval
is changed. Original source wording (including objections and historical
editorial context) is preserved.

`prepare.py` uses the real bundled FP32 Core ML CPU embedder and the existing
retrieval implementation, checks the corpus hash before using fixed chunk
indices, and verifies the shipped model hash. `inputs.json` preserves all A/B/C
jobs and identities; `verified-selection.json` preserves selected corpus records;
`rubric.json` defines assessment criteria. Each completed answer and its exact
system prompt is atomically saved so interrupted experiments can be inspected.
Do not reuse an answers file after changing inputs, prompt, runtime or backend.

## Execution status

The GPU simulator attempt on September 7 failed before generation: this package
requests a 402,653,184-byte tensor allocation, exceeding the simulator GPU's
268,435,456-byte limit. This is an execution failure, not an answer-quality result.
The physical iPhone was unavailable on September 9.

A Debug-only, explicitly selected CPU backend loaded the exact same package on
the arm64 iPhone 17 simulator and completed all 49 jobs on September 9. The
test passed with no runtime errors. See `RESULTS.md` and `assessments.json` for
the human assessment and each answer. CPU results are exploratory and do not
validate physical-device GPU numerical parity, latency, thermals or stability.

GPU log: `/tmp/aquinas-evidence-ablation-xcode.log`.
CPU smoke log: `/tmp/aquinas-evidence-ablation-cpu.log`.
CPU full log: `/tmp/aquinas-evidence-ablation-cpu-full.log`.
Xcode result: `Aquinas-iOS/Test-Aquinas-iOS-2026.09.09_07-35-38--0400.xcresult`.

The test reads the opt-in marker `/tmp/aquinas-evidence-ablation-request.json`.
Remove this marker after running so ordinary test invocations cannot start an
expensive inference run. Only invoke the experiment explicitly:

```sh
xcodebuild -project Aquinas-iOS/Aquinas-iOS.xcodeproj -scheme Aquinas-iOS \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -parallel-testing-enabled NO \
  '-only-testing:Aquinas-iOSTests/EvidenceAblationTests/compareEvidence()' test
```

Run from the shared Developer directory. The parentheses in the Swift Testing
function identifier are required; omitting them can select zero tests.
