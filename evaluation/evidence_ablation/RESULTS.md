# Evidence ablation results — 2026-09-09

This is an exploratory CPU-simulator experiment using the exact shipped Gemma
package, its production conversation prompt, deterministic decoding, and empty
conversation history. It measures how the runtime uses supplied evidence. It
does not measure end-to-end app behavior, device GPU behavior, or latency.

## Result

| Condition | Correct | Questions |
| --- | ---: | ---: |
| Automatic retrieval (A) | 7 / 17 | 41% |
| Verified corpus chunks (B) | 9 / 15 | 60% |
| No passages (C) | 9 / 17 | 53% |

The two extra A/C questions are deliberately ungrounded: the current pope and
Vatican II. The model wrongly named Pope Francis as current in both conditions;
it happened to give Vatican II's correct dates without source evidence. For the
15 answerable questions only, A was 6/15 (40%), B 9/15 (60%), and C 8/15 (53%).

The most diagnostic subset is the seven conciliar/creedal history questions:

| Condition | Correct |
| --- | ---: |
| Automatic retrieval (A) | 2 / 7 |
| Verified corpus chunks (B) | 6 / 7 |
| No passages (C) | 3 / 7 |

## What this establishes

The new sources contain enough material to improve these questions: moving
from automatic retrieval to verified whole chunks changes history from 2/7 to
6/7. The immediate bottleneck is therefore retrieval/ranking and evidence
selection, especially for councils and creeds.

It also establishes a separate generation limit. Even with the right passage,
the model sometimes treated an objection or editorial comment as doctrine,
misattributed an author, or reversed a distinction. It failed all three lying
conditions, including the verified Aquinas passage. More source text alone will
not make every answer reliable.

The no-passage answers being correct 8/15 show that the model has useful stored
knowledge, but it is not a reliable grounding mechanism: it confused the
resurrection accounts, gave a stale current-pope answer, and did not signal when
it lacked evidence.

## Assessment method and limits

Each response was judged against the answer-specific criteria in `rubric.json`.
`assessments.json` records each decision and material error. The assessment is
deliberately strict: a materially wrong council date, creed content, source
attribution, or Thomistic moral conclusion fails the response even when it has
some correct surrounding explanation. This is a 49-response diagnostic sample,
not a statistical benchmark.

The simulator GPU cannot load this model package because one allocation exceeds
its 256 MiB GPU allocation cap. A physical device was unavailable, so this run
used the exact package through the CPU backend. It proves neither GPU numerical
parity nor device performance.

## Recommended next experiment

Hold the model and corpus fixed. Add retrieval query expansion/reranking for
the named councils and creeds, then repeat the A condition against this same
fixture. The pass criterion is A approaching B on the seven history questions;
only then should broader generation or answer-policy work be evaluated.
