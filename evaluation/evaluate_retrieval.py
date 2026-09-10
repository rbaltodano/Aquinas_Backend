#!/usr/bin/env python3
"""Measure whether on-device grounding retrieves passages that actually contain the answer.

This exists because "retrieval returned real ranked passages" is not the same claim as
"retrieval returned the *right* passages", and until now the second claim rested entirely on
hand-picked anecdotes. It scores a fixed case set against the real bundled corpus and the real
bundled Core ML model, so the number moves only when retrieval quality actually moves.

It deliberately mirrors the iOS pipeline (Aquinas-iOS/Services/MiniLMGroundingProvider.swift):
citation lookup against the corpus's own [JHN14] chapter tags, then semantic search under the
same relevance floor. The curated hand-written layer is OFF by default -- curated notes only fire
on questions someone already anticipated, so including them flatters the score on exactly the
questions that need no help. Run with --with-curated to see how much of the score depends on them.

The book-alias table and curated entries are parsed out of the Swift sources at runtime rather
than duplicated here, so this cannot silently drift from what ships.

Usage:
    python evaluation/evaluate_retrieval.py [--with-curated] [--limit 3] [--floor 0.55] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
IOS_ROOT = BACKEND_ROOT.parent / "Aquinas-iOS"
GROUNDING_DIR = IOS_ROOT / "Aquinas-iOS" / "LocalGrounding"
SWIFT_CITATION = IOS_ROOT / "Aquinas-iOS" / "Services" / "ScriptureCitation.swift"
SWIFT_CURATED = IOS_ROOT / "Aquinas-iOS" / "Services" / "AquinasGrounding.swift"
SWIFT_SOURCE_ROUTING = IOS_ROOT / "Aquinas-iOS" / "Services" / "MiniLMGroundingProvider.swift"
CASES = Path(__file__).resolve().parent / "retrieval_cases.json"

EMBEDDING_DIM = 384
SEQUENCE_LENGTH = 128


def load_corpus():
    passages = json.loads((GROUNDING_DIR / "passages.json").read_text())
    embeddings = np.fromfile(GROUNDING_DIR / "embeddings.bin", dtype=np.float32)
    embeddings = embeddings.reshape(len(passages), EMBEDDING_DIM)
    return passages, embeddings


def load_embedder():
    """The bundled Core ML model, matching what the phone runs. Uses CPU_ONLY deliberately: the
    iOS Simulator forces that path, and an FP16 export makes it return garbage there (see
    export_minilm_coreml.py), so scoring it here keeps that regression visible."""
    import coremltools as ct
    from transformers import AutoTokenizer

    model = ct.models.MLModel(
        str(GROUNDING_DIR / "MiniLM.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def embed(text: str) -> np.ndarray:
        encoded = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=SEQUENCE_LENGTH, return_tensors="np",
        )
        output = model.predict({
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32),
        })
        vector = np.array(list(output.values())[0]).reshape(-1).astype(np.float32)
        if not np.all(np.isfinite(vector)):
            raise SystemExit(
                "The bundled Core ML model returned non-finite output on the CPU path. "
                "Re-export it at FP32 -- see scripts/export_minilm_coreml.py."
            )
        return vector

    return embed


def parse_book_aliases() -> list[tuple[str, str]]:
    """Read the (alias, USFM code, display) table out of ScriptureCitation.swift."""
    source = SWIFT_CITATION.read_text()
    pairs = re.findall(r'\("([^"]+)",\s*"([A-Z0-9]{3})",\s*"[^"]*"\)', source)
    if not pairs:
        raise SystemExit(f"Could not parse book aliases from {SWIFT_CITATION}")
    return sorted(pairs, key=lambda p: -len(p[0]))


def parse_named_passages() -> list[tuple[str, str, int]]:
    """Read the named-passage table ("the Beatitudes" -> MAT 5) out of ScriptureCitation.swift."""
    source = SWIFT_CITATION.read_text()
    found = re.findall(r'\("([^"]+)",\s*"([A-Z0-9]{3})",\s*(\d+),\s*"[^"]*"\)', source)
    return [(name, code, int(chapter)) for name, code, chapter in found]


def parse_named_passage_anchors() -> dict[str, str]:
    """Read literal named-passage anchors out of ScriptureCitation.swift."""
    source = SWIFT_CITATION.read_text()
    block = re.search(r'namedPassageAnchors: \[String: String\] = \[(.*?)\n    \]', source, re.S)
    if not block:
        return {}
    return dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', block.group(1)))


def parse_curated() -> list[dict]:
    """Read curated reference entries out of AquinasGrounding.swift."""
    source = SWIFT_CURATED.read_text()
    entries = []
    for block in re.findall(r"AquinasGroundingReference\((.*?)\n        \)", source, re.S):
        facts = re.search(r'facts:\s*"((?:[^"\\]|\\.)*)"', block)
        aliases = re.search(r"retrievalAliases:\s*\[(.*?)\]", block, re.S)
        title = re.search(r'title:\s*"([^"]*)"', block)
        if not (facts and aliases and title):
            continue
        entries.append({
            "title": title.group(1),
            "facts": facts.group(1).encode().decode("unicode_escape"),
            "aliases": re.findall(r'"([^"]+)"', aliases.group(1)),
        })
    return entries


def parse_named_sources() -> list[tuple[str, set[str]]]:
    """Read document-name routing aliases from MiniLMGroundingProvider.swift.

    Keeping this table in Swift makes the shipped app the authority. Parsing it here
    ensures the evaluator measures precisely that behavior instead of a drifting copy.
    """
    source = SWIFT_SOURCE_ROUTING.read_text()
    entries = re.findall(r'\("([^"]+)",\s*\[([^\]]+)\]\)', source)
    if not entries:
        raise SystemExit(f"Could not parse named source aliases from {SWIFT_SOURCE_ROUTING}")
    return [
        (name, set(re.findall(r'"([^"]+)"', source_ids)))
        for name, source_ids in entries
    ]


def parse_authority_sections() -> list[tuple[set[str], set[str], set[str]]]:
    """Read section-level authority pointers from the iOS retrieval provider.

    These are corpus locations, not answer text. Parsing the shipping table here keeps the
    evaluator aligned with the app whenever a named doctrinal question gains a precise pointer.
    """
    source = SWIFT_SOURCE_ROUTING.read_text()
    entries = re.findall(
        r'authorityTerms:\s*\[([^\]]+)\],\s*topicTerms:\s*\[([^\]]+)\],\s*sourceIDs:\s*\[([^\]]+)\],\s*sectionTerms:\s*\[([^\]]+)\]',
        source,
        re.S,
    )
    return [
        (
            set(re.findall(r'"([^"]+)"', authority_terms)),
            set(re.findall(r'"([^"]+)"', topic_terms)),
            set(re.findall(r'"([^"]+)"', source_ids)),
            set(re.findall(r'"([^"]+)"', section_terms)),
        )
        for authority_terms, topic_terms, source_ids, section_terms in entries
    ]


def named_source_search_terms(question: str, named_sources) -> set[str]:
    """Mirror NamedCorpusSource.searchTerms(in:) in the iOS provider."""
    stop_words = {
        "about", "after", "against", "and", "before", "could", "does", "from", "have", "into",
        "are", "council", "catechism", "decide", "did", "does", "is", "say", "should", "teach", "that", "the", "their", "these", "they", "this", "was", "what", "when", "who", "why",
        "where", "which", "with", "would",
    }
    terms = {
        word for word in re.findall(r"[a-z0-9]+", question.casefold())
        if len(word) >= 3 and word not in stop_words
    }
    matching_alias_words = {
        word
        for alias, _ in named_sources
        if alias in question.casefold()
        for word in re.findall(r"[a-z0-9]+", alias)
    }
    topic_terms = terms - matching_alias_words
    # Mirror NamedCorpusSource.personAliases. A person's name can be the whole lookup term, while
    # a document title should preserve semantic source ranking to avoid front matter winning.
    matches_person_alias = any(alias == "arius" and alias in question.casefold()
                               for alias, _ in named_sources)
    return terms if not topic_terms and matches_person_alias else topic_terms


def named_source_ranking_query(question: str, named_sources) -> str:
    """Mirror NamedCorpusSource.rankingQuery(in:) in the iOS provider."""
    terms = sorted(named_source_search_terms(question, named_sources))
    return " ".join(terms) or question


def index_chapters(passages) -> dict[str, range]:
    """Mirror OnDeviceGroundingStore.indexChapters."""
    tag = re.compile(r"^\s*\[([A-Z0-9]{3,8})\]")
    def key_of(text):
        match = re.match(r"^\s*\[([A-Z0-9]{3})(\d+)\]", text.lstrip("﻿"))
        return f"{match.group(1)}{int(match.group(2))}" if match else None

    starts = [(key_of(p["text"]), i) for i, p in enumerate(passages)]
    starts = [(k, i) for k, i in starts if k]
    ranges: dict[str, range] = {}
    for offset, (key, index) in enumerate(starts):
        source_id = passages[index]["sourceId"]
        end = index + 1
        while end < len(passages) and passages[end]["sourceId"] == source_id:
            if offset + 1 < len(starts) and starts[offset + 1][1] == end:
                break
            if tag.match(passages[end]["text"].lstrip("﻿")):
                break
            end += 1
        ranges.setdefault(key, range(index, end))
    return ranges


def citations_in(question: str, aliases, named=(), anchors=None) -> list[tuple[str, int, str | None]]:
    """Mirror ScriptureCitation.citations(in:)."""
    text = question.lower()
    found, claimed = [], []
    anchors = anchors or {}
    for name, code, chapter in sorted(named, key=lambda n: -len(n[0])):
        if name in text and (code, chapter) not in [(c, ch) for c, ch, _ in found]:
            found.append((code, chapter, anchors.get(name)))
    for alias, code in aliases:
        for match in re.finditer(re.escape(alias), text):
            start, end = match.span()
            if start > 0 and text[start - 1].isalpha():
                continue
            if end < len(text) and text[end].isalpha():
                continue
            if any(s < end and start < e for s, e in claimed):
                continue
            tail = re.match(r"[\s.,]*(?:chapter|ch )?[\s.,]*(\d{1,3})", text[end:])
            if not tail:
                continue
            chapter = int(tail.group(1))
            if not 0 < chapter < 200:
                continue
            claimed.append((start, end + tail.end()))
            found.append((code, chapter, None))
    return found


def outside_corpus_scope(question: str) -> bool:
    """Mirror CorpusScope.excludes in the shipped iOS provider."""
    normalized = question.casefold()
    words = set(re.findall(r"[a-z0-9]+", normalized))
    return (("current" in words and "pope" in words)
            or "vatican ii" in normalized
            or "vatican 2" in normalized
            or "second vatican council" in normalized)


def retrieve(question, *, embed, passages, embeddings, chapters, book_aliases,
             named_passages, named_passage_anchors, named_sources, authority_sections, curated, limit, floor,
             corroboration_floor=0.62):
    """Mirror MiniLMGroundingProvider.references(for:limit:)."""
    if outside_corpus_scope(question):
        return []
    collected: list[dict] = []

    if curated is not None:
        scored = []
        for entry in curated:
            score = sum(len(a) for a in entry["aliases"] if a.lower() in question.lower())
            if score:
                scored.append((score, entry))
        for _, entry in sorted(scored, key=lambda s: -s[0])[: max(1, limit - 1)]:
            collected.append({"layer": "curated", "title": entry["title"], "text": entry["facts"],
                              "score": None})

    for code, chapter, anchor in citations_in(question, book_aliases, named_passages, named_passage_anchors):
        if len(collected) >= limit:
            break
        chapter_indices = chapters.get(f"{code}{chapter}", [])
        if anchor:
            anchor_index = next(
                (index for index in chapter_indices if anchor in passages[index]["text"].casefold()),
                None,
            )
            # Mirror the app: a stale source-text pointer returns nothing rather than falling back
            # to an unrelated opening chunk from the same chapter.
            if anchor_index is None:
                continue
            chapter_indices = range(anchor_index, chapter_indices.stop)
        for index in chapter_indices[: limit - len(collected)]:
            collected.append({"layer": "citation", "title": passages[index]["title"],
                              "text": passages[index]["text"], "score": None})

    # A section pointer identifies the primary-source passage by its own heading or formula. It
    # is a metadata lookup, comparable to a Bible chapter citation, not a generated answer.
    question_folded = question.casefold()
    question_words = set(re.findall(r"[a-z0-9]+", question_folded))
    if len(collected) < limit:
        for authority_terms, topic_terms, source_ids, section_terms in authority_sections:
            if not authority_terms.issubset(question_words) or topic_terms.isdisjoint(question_words):
                continue
            for index, passage in enumerate(passages):
                text = passage["text"].casefold()
                if (passage["sourceId"] in source_ids
                        and all(term in text for term in section_terms)):
                    # Mirror OnDeviceGroundingStore.section: a pointer anchors a source-local
                    # run so the prompt contains the explanation following a short heading.
                    source_id = passage["sourceId"]
                    for section_index in range(index, len(passages)):
                        section_passage = passages[section_index]
                        if section_passage["sourceId"] != source_id or len(collected) >= limit:
                            break
                        collected.append({"layer": "authority-section", "title": section_passage["title"],
                                          "text": section_passage["text"], "score": None})
                    break
            if len(collected) >= limit:
                break

    # A named council, creed, or work title identifies a document rather than a broad
    # historical topic. Mirror the app by ranking within that actual source first.
    # This adds only exported corpus passages; it never supplies a written answer.
    if len(collected) < limit:
        source_ids = {
            source_id
            for alias, source_set in named_sources
            if alias in question_folded
            for source_id in source_set
        }
        if source_ids:
            similarities = embeddings @ embed(named_source_ranking_query(question, named_sources))
            prioritizing_terms = named_source_search_terms(question, named_sources)
            candidate_indices = [
                index for index, passage in enumerate(passages)
                if passage["sourceId"] in source_ids
                and (
                    similarities[index] >= floor
                    or any(term in passage["text"].casefold() for term in prioritizing_terms)
                )
            ]
            def source_rank(index):
                text = passages[index]["text"].casefold()
                matched_terms = sum(term in text for term in prioritizing_terms)
                return (-matched_terms, -similarities[index])

            for index in sorted(candidate_indices, key=source_rank):
                if len(collected) >= limit:
                    break
                if any(reference["text"] == passages[index]["text"] for reference in collected):
                    continue
                collected.append({"layer": "source", "title": passages[index]["title"],
                                  "text": passages[index]["text"],
                                  "score": float(similarities[index])})

    if len(collected) < limit:
        # Mirror the tiered floor in MiniLMGroundingProvider: a semantic passage standing on its
        # own only needs the standard floor, but one merely padding an already-authoritative
        # curated fact or cited chapter has to clear a stricter bar.
        effective_floor = floor if not collected else corroboration_floor
        similarities = embeddings @ embed(question)
        for index in np.argsort(-similarities)[: limit - len(collected)]:
            if similarities[index] < effective_floor:
                break
            if any(reference["text"] == passages[index]["text"] for reference in collected):
                continue
            collected.append({"layer": "semantic", "title": passages[index]["title"],
                              "text": passages[index]["text"],
                              "score": float(similarities[index])})
    return collected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-curated", action="store_true",
                        help="include the hand-written curated layer (off by default)")
    parser.add_argument("--limit", type=int, default=3, help="passages per question (app uses 3)")
    parser.add_argument("--floor", type=float, default=0.55,
                        help="similarity floor; the app's 0.45 max distance")
    parser.add_argument("--corroboration-floor", type=float, default=0.62,
                        help="stricter floor for semantic passages added alongside an "
                             "authoritative curated fact or cited chapter")
    parser.add_argument("--json", type=Path, help="write the full result to this path")
    parser.add_argument("--verbose", action="store_true", help="print every retrieved passage")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text())
    passages, embeddings = load_corpus()
    embed = load_embedder()
    chapters = index_chapters(passages)
    book_aliases = parse_book_aliases()
    named_passages = parse_named_passages()
    named_passage_anchors = parse_named_passage_anchors()
    named_sources = parse_named_sources()
    authority_sections = parse_authority_sections()
    curated = parse_curated() if args.with_curated else None

    print(f"{len(cases)} cases | {len(passages):,} passages | floor {args.floor} "
          f"| curated layer {'ON' if args.with_curated else 'OFF'}\n")

    results, by_category = [], defaultdict(lambda: {"pass": 0, "total": 0})
    for case in cases:
        refs = retrieve(case["question"], embed=embed, passages=passages, embeddings=embeddings,
                        chapters=chapters, book_aliases=book_aliases,
                        named_passages=named_passages, named_passage_anchors=named_passage_anchors,
                        named_sources=named_sources,
                        authority_sections=authority_sections, curated=curated,
                        limit=args.limit, floor=args.floor,
                        corroboration_floor=args.corroboration_floor)
        grounded = bool(refs)
        blob = "\n".join(r["text"] for r in refs).lower()
        hit = any(p.lower() in blob for p in case["evidence_any"])

        if case["expect"] == "abstain":
            passed, failure = not grounded, None if not grounded else "grounded-when-it-should-abstain"
        elif not grounded:
            passed, failure = False, "no-coverage"
        elif not hit:
            passed, failure = False, "wrong-passages"
        else:
            passed, failure = True, None

        by_category[case["category"]]["total"] += 1
        by_category[case["category"]]["pass"] += int(passed)
        results.append({**case, "passed": passed, "failure": failure,
                        "layers": [r["layer"] for r in refs],
                        "top_score": next((r["score"] for r in refs if r["score"]), None),
                        "titles": [r["title"] for r in refs]})

        if args.verbose or not passed:
            mark = "PASS" if passed else "FAIL"
            print(f"[{mark}] {case['id']}" + (f"  ({failure})" if failure else ""))
            for ref in refs:
                score = f"{ref['score']:.3f}" if ref["score"] else "  -  "
                print(f"        {score} [{ref['layer']}] {ref['title']}: {ref['text'][:70]}"
                      .replace("\n", " "))

    total_pass = sum(r["passed"] for r in results)
    print(f"\n{'=' * 62}\nOVERALL  {total_pass}/{len(results)}  ({total_pass / len(results):.0%})\n")
    print(f"{'category':<16}{'pass':>10}   rate")
    for name, stat in sorted(by_category.items()):
        print(f"{name:<16}{stat['pass']:>4}/{stat['total']:<5}   {stat['pass'] / stat['total']:.0%}")

    modes = defaultdict(int)
    for result in results:
        if result["failure"]:
            modes[result["failure"]] += 1
    if modes:
        print("\nfailure modes")
        for name, count in sorted(modes.items(), key=lambda m: -m[1]):
            print(f"  {count:>3}  {name}")
        print("\n  no-coverage    = corpus has nothing; model answers unguided (ingest sources)")
        print("  wrong-passages = retrieved confident but irrelevant text (the dangerous one)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"summary": {"passed": total_pass, "total": len(results),
                         "with_curated": args.with_curated, "floor": args.floor},
             "by_category": dict(by_category), "cases": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
