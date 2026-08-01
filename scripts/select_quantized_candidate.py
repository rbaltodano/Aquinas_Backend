"""Select the fastest reviewed checkpoint that satisfies Aquinas quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def candidate_summary(candidate: list[dict], baseline_p50: float) -> dict:
    reviewed = [
        record["human_review"]["no_worse_than_bf16"]
        for record in candidate
        if record["human_review"]["no_worse_than_bf16"] is not None
    ]
    critical_regression = any(
        record["human_review"]["critical_regression"] is True
        for record in candidate
    )
    p50 = statistics.median(record["total_seconds"] for record in candidate)
    parity = (
        sum(value is True for value in reviewed) / len(reviewed)
        if reviewed
        else 0.0
    )
    passes = (
        len(reviewed) == len(candidate)
        and all(record["valid"] for record in candidate)
        and not critical_regression
        and parity >= 0.90
        and p50 <= baseline_p50 * 0.80
    )
    return {
        "passes": passes,
        "p50_total_seconds": p50,
        "quality_parity": parity,
        "fully_reviewed": len(reviewed) == len(candidate),
        "critical_regression": critical_regression,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--six-bit", type=Path, required=True)
    parser.add_argument("--four-bit", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_results(args.bf16)
    baseline_p50 = statistics.median(
        record["total_seconds"]
        for record in baseline
    )
    summaries = {
        "6bit": candidate_summary(load_results(args.six_bit), baseline_p50),
        "4bit": candidate_summary(load_results(args.four_bit), baseline_p50),
    }
    passing = [
        (name, summary)
        for name, summary in summaries.items()
        if summary["passes"]
    ]
    selected = (
        min(passing, key=lambda item: item[1]["p50_total_seconds"])[0]
        if passing
        else "bf16"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "bf16_p50_total_seconds": baseline_p50,
                "candidates": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
