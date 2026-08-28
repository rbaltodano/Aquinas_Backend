"""Rebuilds the Aquinas fine-tuning dataset from the raw Summa Theologica text.

Supersedes the old finalize_training_data.py + convert_gemma.py pipeline, which
produced a corrupted dataset: the raw source is a PDF-to-text extraction that
repeats a running page header once per page -- the current article's title,
truncated with an ellipsis ("..."). The old split regex treated every
occurrence of "Article. N -" as a new article boundary, including these
repeated headers, so ~33% of the resulting training examples were random
mid-article fragments mislabeled as complete articles (confirmed by their
sizes clustering tightly around ~3,200 characters -- one PDF page -- and by
manual inspection of the raw source around a truncation point).

A repeated page header always cites the SAME article number as whichever
article is currently in progress (it's a running header for that article,
repeated once per page until the article ends). A genuine new article
heading is never immediately preceded by a heading with that same number.
That adjacency check is what's used here to strip the noise before
splitting -- it doesn't depend on title length or truncation, unlike an
ellipsis-based heuristic (tried first; left ~31% of examples still
fragmented because short titles fit on one line without truncation and so
never got an ellipsis).
"""

import json
import os
import random
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
SOURCE_PATH = os.path.join(RAW_DATA_DIR, "summa_theologica.txt")
OUTPUT_DIR = os.path.join(DATA_DIR, "training_data")

PROMPT = "Explain this Scholastic concept using the method of the Summa."

ARTICLE_HEADING_LINE = re.compile(r'^\s*Article\.?\s+(\d+)\s*-.*$', re.MULTILINE)
ARTICLE_SPLIT_PATTERN = re.compile(
    r"(Article\.?\s+\d+\s*-\s*.*?)(?=Article\.?\s+\d+\s*-|\Z)", re.DOTALL
)


def clean_source(raw_text: str) -> str:
    # Strip lone page-number lines.
    text = re.sub(r'^\s*\d+\s*$', '', raw_text, flags=re.MULTILINE)

    # Strip repeated running-page-header lines: a heading whose article number
    # matches the immediately preceding heading's number is a duplicate running
    # header, not a real new article, regardless of whether its title happened
    # to be truncated with an ellipsis.
    out = []
    last_end = 0
    prev_number = None
    for m in ARTICLE_HEADING_LINE.finditer(text):
        number = m.group(1)
        if number == prev_number:
            out.append(text[last_end:m.start()])
            last_end = m.end()
            continue
        prev_number = number
    out.append(text[last_end:])
    return "".join(out)


def split_articles(clean_text: str) -> list[str]:
    matches = ARTICLE_SPLIT_PATTERN.findall(clean_text)
    articles = []
    for match in matches:
        full_text = " ".join(line.strip() for line in match.split("\n") if line.strip())
        full_text = full_text.strip()
        if full_text:
            articles.append(full_text)
    return articles


def main():
    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = clean_source(raw_text)
    articles = split_articles(clean_text)
    print(f"Split into {len(articles)} article-level examples.")

    # Sanity check: how many still end mid-sentence (no terminal punctuation)?
    mid_sentence = sum(1 for a in articles if a and a[-1] not in '.!?"’”')
    print(
        f"{mid_sentence} of {len(articles)} ({mid_sentence / len(articles) * 100:.1f}%) "
        "still lack terminal punctuation (expect this to be near 0)."
    )

    examples = [
        {"messages": [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": text}]}
        for text in articles
    ]

    random.seed(20260828)
    random.shuffle(examples)

    n = len(examples)
    n_valid = max(1, round(n * 0.09))
    n_test = max(1, round(n * 0.09))
    valid = examples[:n_valid]
    test = examples[n_valid:n_valid + n_test]
    train = examples[n_valid + n_test:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, split in [("train.jsonl", train), ("valid.jsonl", valid), ("test.jsonl", test)]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split)} examples to {path}")


if __name__ == "__main__":
    main()
