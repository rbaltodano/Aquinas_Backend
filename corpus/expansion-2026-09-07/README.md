# Conciliar, creedal and catechetical corpus expansion — 2026-09-07

The corpus increased from 48,048 to 50,283 passages. The unchanged 56-case
retrieval evaluation stayed at 46/56 (82.14%): two gains and two regressions.
This is retrieval evidence, not a measurement of generated answer accuracy.

| Category | Before | After |
| --- | --- | --- |
| Church history | 4/7 (57%) | 5/7 (71%) |
| Creeds | 1/2 (50%) | 2/2 (100%) |
| Out of scope | 7/7 (100%) | 5/7 (71%) |
| Doctrine | 12/12 | 12/12 |
| Sacraments | 4/4 | 4/4 |
| Scripture | 9/12 | 9/12 |
| Apologetics | 5/6 | 5/6 |
| Moral | 4/6 | 4/6 |

Gains: `history-nicaea-ii` (defense of icons) and `history-nicene-creed`.
Regressions: `abstain-current-pope` and `abstain-vatican-ii`. Historical council
passages now clear the standard similarity floor on those questions despite
not answering them. Arius and Didache still fail. Adding coverage alone did
not produce a net reliability gain, and the new false grounding is a material
limitation of this export.

No curated entries, retrieval thresholds, eval cases, embedding precision or
generation code changed. Both runs use the bundled Core ML CPU-only path,
three passages, standard similarity 0.55 and corroboration similarity 0.62,
with the curated layer OFF. `before.json` and `after.json` retain every case.

## Sources and rights

All public-domain determinations below are for the USA and the specific
historical English editions, not modern translations or website material.

* **1,853 passages:** Henry R. Percival, *The Seven Ecumenical Councils*, NPNF
  series II, volume XIV (1900). [CCEL's text](https://www.ccel.org/ccel/s/schaff/npnf214/cache/npnf214.txt)
  explicitly records “Rights: Public Domain.” Includes canons, decrees,
  creeds, letters, local synods and historical editorial commentary. The
  source title identifies the commentary; it is not all conciliar teaching.
* **63 passages:** Philip Schaff, *Creeds of Christendom*, volume II, sixth
  edition, Oecumenical Symbols section. [The title page](https://ccel.org/ccel/schaff/creeds2/creeds2.i.html)
  records copyrights through 1919, and CCEL's download identifies it as public
  domain. Includes Apostles', Nicene/Constantinopolitan, Chalcedonian and
  Athanasian formulas with original-language parallels and historical notes.
* **319 passages:** *Baltimore Catechism No. 3*, supplemented by Thomas L.
  Kinkead, edition carrying a 1921 imprimatur. [Gutenberg ebook 14553](https://www.gutenberg.org/ebooks/14553)
  explicitly identifies it as public domain in the USA. Historical discipline
  in this catechism is not necessarily contemporary Catholic discipline.

[CCEL's policy](https://www.ccel.org/about/copyright.html) distinguishes public
domain books from protected website/special contents. The extractor takes
only historical text, omitting distributor headers, generated indexes and
footers; Gutenberg transcriber annotations are also excluded. Missing edition
boundaries raise an error. `receipt.json` preserves source retrieval times,
cleaned-text hashes, and copied asset sizes/hashes. Runtime assets and full
download logs remain gitignored, so the receipt does not replace regeneration.

## Reproduction and verification

From the backend repository:

```sh
./coreml_conversion_env/bin/python evaluation/evaluate_retrieval.py --json evaluation/results/before.json
./aquinas_env/bin/python ingest_corpus.py --ids seven-ecumenical-councils ecumenical-creeds-schaff baltimore-catechism-3
./coreml_conversion_env/bin/python scripts/export_minilm_coreml.py
./aquinas_env/bin/python scripts/export_on_device_grounding.py
```

Copy `MiniLM.mlpackage`, `vocab.txt`, `passages.json`, and `embeddings.bin` from
`data/corpus/on_device_export/` to `../Aquinas-iOS/Aquinas-iOS/LocalGrounding/`,
then run the same evaluation with an after JSON path. The before measurement
must precede replacement of the old bundled assets.

Completed both exports and verified SHA-256 equality for all six constituent
asset files. Embeddings contain exactly 50,283 × 384 × 4 bytes. MiniLM remains
FP32; the default and CPU-only fidelity checks both passed with worst cosine
1.000000 at six decimals. All five ingestion unit tests passed.
All 48,048 pre-existing passage records and their float32 vectors are unchanged
byte-for-byte after matching on source ID and chunk index; all new vectors are
finite.

The requested iPhone 17 simulator test command built and signed the app and
reported 103 cases (99 passed, four failed), then stalled after the test host
exited; the runner was interrupted and terminated, so there is no clean full
suite completion. Failures were the two previously documented
`thinkingSummary*` cases, `localGroundingRetrievesJohn14`, and
`shortPauseKeepsModelWarm`. The John 14 test incorrectly required the removed
negative example "John 4"; its assertion now requires that phrase to be absent.
No grounding note was changed. Both that corrected test and the unchanged
lifecycle test passed in a focused run (two Swift tests actually executed),
with `-parallel-testing-enabled NO` and full `-only-testing` function names
including `()`. The lifecycle failure did not reproduce in isolation.
The focused result is
`~/Library/Developer/Xcode/DerivedData/Aquinas-iOS-fhnimhcihqeolrbcuqajrnnpkcis/Logs/Test/Test-Aquinas-iOS-2026.09.07_17-16-23--0400.xcresult`.
Full-run and focused logs are `/tmp/aquinas-corpus-expansion-xcode-test.log`
and `/tmp/aquinas-corpus-expansion-focused-test.log` respectively.

The pre-expansion local assets were backed up at
`/tmp/aquinas-grounding-before-20260907`; that temporary backup is not a
durable release artifact.
