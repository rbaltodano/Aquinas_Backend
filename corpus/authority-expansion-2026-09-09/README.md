# Authority corpus expansion — 2026-09-09

The on-device corpus increased from 50,283 to 51,836 passages.

| Source | Historical English edition | Passages |
| --- | --- | ---: |
| *Canons and Decrees of the Council of Trent* | J. Waterworth, 1848 | 559 |
| *Catechism of the Council of Trent* | Jeremiah Donovan, 1829 | 994 |

Both editions are public domain in the United States. The Waterworth scan is
identified as an 1848 English edition by its [Internet Archive record](https://archive.org/details/TheCanonsAndDecrees).
The Donovan edition's [Wikimedia source record](https://commons.wikimedia.org/wiki/File:Catechism_of_the_Council_of_Trent_(IA_CatechismOfTheCouncilOfTrent1829).pdf)
identifies its 1829 publication and US public-domain status. Source boundaries
exclude Waterworth's historical essays and index, and Donovan's front matter,
translator's preface and index.

## Verification

`evaluation/evaluate_retrieval.py` was run with the real bundled FP32 Core ML
model on CPU only, curated layer off, before the additions and after the final
export. Both scores were **46/56 (82%)**. Every category count was unchanged.
The fixed evaluation has no Council of Trent or Roman Catechism case, so it
cannot measure direct coverage gains from these sources. It does show that this
addition did not improve any existing case and did not create an additional
regression.

`scripts/export_minilm_coreml.py` was rerun after the final ingestion. Its
reference and CPU-only checks both reported worst cosine similarity **1.000000**;
the MiniLM package remains FP32. `scripts/export_on_device_grounding.py` wrote
51,836 passages and a corresponding float32 embedding matrix. The copied
`passages.json` and `embeddings.bin` in `Aquinas-iOS/LocalGrounding/` matched
the export SHA-256 values exactly.

## Follow-up: named-source routing

The follow-up added two direct authority cases (Trent on justification and the
Roman Catechism), then measured source routing against the real bundled FP32
model. Exact work and council names select their source first; meaningful terms
in the question select the relevant passage within it. This is corpus retrieval,
not a curated response, and it leaves the global 0.45 and corroboration 0.38
distance floors unchanged.

| Evaluation | Overall | Church history |
| --- | ---: | ---: |
| Current direct-evidence routing check, 65 cases | 55/65 (85%) | 8/10 (80%) |

The original authority check accepted a Trent chapter heading as evidence. It
was strengthened to require the actual Session VI definition ("not remission
of sins merely"). The current result is therefore not comparable to that
earlier, weaker 50/58 measurement. The evaluator reads the source-alias table from
`Aquinas-iOS/Services/MiniLMGroundingProvider.swift`, preventing drift between
the shipped behavior and the measurement. Results are retained in
`evaluation/results/authority-cases-before-routing.json` and
`evaluation/results/authority-cases-after-routing.json`.

### Section pointers

The app also carries a small source-location table for questions that name an
authority and a doctrine together. It selects the matching passage by its own
heading or formula before broad source ranking: Trent/justification selects
Session VI, Chapter VII; Nicaea/the Son selects the creed; Chalcedon/two
natures selects its Definition. These entries contain only source identifiers
and text already present in the exported corpus, never an answer paraphrase.
The evaluator parses the table from the iOS provider and confirms those primary
formulations are retrieved. Natural-language cases for Nicaea/Christ,
Chalcedon/Christ, Trent/Eucharist and penance, and the Roman Catechism on
Baptism, Eucharist, and penance all pass with the pointers, producing the
55/65 result. A pointer also retrieves its following source-local chunks, so a
brief heading is accompanied by the primary text's explanation rather than
standing alone in the prompt.
