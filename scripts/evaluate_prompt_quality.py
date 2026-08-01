"""Run end-to-end Aquinas prompt-quality cases and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from prompt_quality import (
    SUPPORTED_OPERATIONS,
    create_human_review,
    evaluate_checks,
    load_cases,
    objective_passed,
    render_markdown_report,
    summarize_records,
)


class FirstOutputProbe:
    def __init__(self, output: str | None) -> None:
        self.output = output
        self.used = False

    def take(self) -> str | None:
        if self.output is None or self.used:
            return None
        self.used = True
        return self.output


class CountedGenerator:
    def __init__(
        self,
        delegate: Callable[[str, int], str],
        first_output: str | None = None,
        probe: FirstOutputProbe | None = None,
    ) -> None:
        self.delegate = delegate
        self.probe = probe or FirstOutputProbe(first_output)
        self.calls = 0
        self.outputs: list[str] = []

    def __call__(self, prompt: str, max_tokens: int) -> str:
        self.calls += 1
        output = self.probe.take()
        if output is None:
            output = self.delegate(prompt, max_tokens)
        self.outputs.append(output)
        return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/Aquinas-Final",
        help="MLX checkpoint to evaluate.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/prompt_quality_cases.json"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("evaluation/prompt_quality_reviews.json"),
        help="Optional versioned human-review catalog applied to matching cases.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--operation",
        action="append",
        choices=sorted(SUPPORTED_OPERATIONS),
        help="Restrict the run to one or more operations.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Restrict the run to one or more exact case IDs.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve existing records and skip completed case IDs.",
    )
    parser.add_argument(
        "--rerun-passing",
        action="store_true",
        help="With --resume, rerun selected objective passes instead of skipping them.",
    )
    parser.add_argument(
        "--skip-minilm",
        action="store_true",
        help="Do not calculate the selected Midpoint candidate.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the catalog without loading either model.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild Markdown and summary from an existing reviewed JSON result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.render_only:
        if args.output is None or not args.output.exists():
            raise SystemExit("--render-only requires an existing --output JSON file.")
        report_path = args.report or args.output.with_suffix(".md")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        refresh_objective_checks(
            payload.get("records", []),
            load_cases(args.cases),
        )
        apply_human_reviews(
            payload.get("records", []),
            load_review_catalog(args.reviews),
        )
        payload["summary"] = summarize_records(payload.get("records", []))
        write_reports(args.output, report_path, payload)
        print(json.dumps(payload["summary"], indent=2))
        return
    cases = load_cases(args.cases)
    cases = filter_cases(cases, args.operation, args.case_id, args.limit)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "cases": len(cases),
                    "operations": sorted({case["operation"] for case in cases}),
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        raise SystemExit("--output is required unless --validate-only is used.")
    report_path = args.report or args.output.with_suffix(".md")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["AQUINAS_MODEL_PATH"] = args.model_path
    from main import (
        generate_aquinas,
        generate_aquinas_background,
        generate_aquinas_background_fast,
        generate_aquinas_fast,
    )

    records = load_resumable_records(args.output) if args.resume else []
    completed_ids = (
        set()
        if args.rerun_passing
        else passing_record_ids(records)
    )
    for case in cases:
        if case["id"] in completed_ids:
            continue
        started_at = time.perf_counter()
        probe = FirstOutputProbe(case.get("repair_probe_output"))
        deep = CountedGenerator(generate_aquinas, probe=probe)
        fast = CountedGenerator(generate_aquinas_fast, probe=probe)
        background = CountedGenerator(generate_aquinas_background, probe=probe)
        background_fast = CountedGenerator(
            generate_aquinas_background_fast,
            probe=probe,
        )
        try:
            output = run_case(
                case,
                deep=deep,
                fast=fast,
                background=background,
                background_fast=background_fast,
                with_minilm=not args.skip_minilm,
            )
            generation_calls = (
                deep.calls
                + fast.calls
                + background.calls
                + background_fast.calls
            )
            checks = evaluate_checks(case, output, generation_calls)
            record = {
                "id": case["id"],
                "operation": case["operation"],
                "description": case.get("description", ""),
                "tags": case.get("tags", []),
                "status": "completed",
                "duration_seconds": time.perf_counter() - started_at,
                "generation_calls": generation_calls,
                "repair_attempted": generation_calls > 1,
                "checks": checks,
                "objective_passed": objective_passed(checks),
                "output": output,
                "human_review": create_human_review(case),
            }
        except Exception as error:
            record = {
                "id": case["id"],
                "operation": case["operation"],
                "description": case.get("description", ""),
                "tags": case.get("tags", []),
                "status": "error",
                "duration_seconds": time.perf_counter() - started_at,
                "generation_calls": (
                    deep.calls
                    + fast.calls
                    + background.calls
                    + background_fast.calls
                ),
                "repair_attempted": (
                    deep.calls
                    + fast.calls
                    + background.calls
                    + background_fast.calls
                ) > 1,
                "checks": [],
                "objective_passed": False,
                "output": None,
                "error": f"{type(error).__name__}: {error}",
                "diagnostic_outputs": diagnostic_outputs(
                    deep,
                    fast,
                    background,
                    background_fast,
                ),
                "human_review": create_human_review(case),
            }
        records = [
            existing
            for existing in records
            if existing.get("id") != record["id"]
        ] + [record]
        apply_human_reviews(records, load_review_catalog(args.reviews))
        payload = build_payload(args.model_path, args.cases, records)
        write_reports(args.output, report_path, payload)
        print(
            json.dumps(
                {
                    "id": record["id"],
                    "status": record["status"],
                    "objective_passed": record["objective_passed"],
                    "duration_seconds": round(record["duration_seconds"], 3),
                }
            )
        )

    apply_human_reviews(records, load_review_catalog(args.reviews))
    payload = build_payload(args.model_path, args.cases, records)
    write_reports(args.output, report_path, payload)
    print(json.dumps(payload["summary"], indent=2))


def filter_cases(
    cases: list[dict[str, Any]],
    operations: list[str] | None,
    case_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = [
        case
        for case in cases
        if (not operations or case["operation"] in operations)
        and (not case_ids or case["id"] in case_ids)
    ]
    return selected[: max(0, limit)] if limit is not None else selected


def load_resumable_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def passing_record_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        record["id"]
        for record in records
        if (
            record.get("status") == "completed"
            and record.get("objective_passed") is True
        )
    }


def load_review_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    reviews = payload.get("reviews", {})
    if not isinstance(reviews, dict):
        raise ValueError("The human-review catalog must contain a reviews object.")
    return reviews


def apply_human_reviews(
    records: list[dict[str, Any]],
    reviews: dict[str, Any],
) -> None:
    for record in records:
        review = reviews.get(record.get("id"))
        if not isinstance(review, dict):
            continue
        score_values = review.get("scores", {})
        if not isinstance(score_values, dict):
            raise ValueError(
                f"Review scores for {record.get('id')} must be an object."
            )
        human_review = record.get("human_review", {})
        for score in human_review.get("scores", []):
            rubric_id = score.get("id")
            if rubric_id in score_values:
                value = score_values[rubric_id]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        f"Review score {record.get('id')}/{rubric_id} "
                        "must be an integer."
                    )
                score["score"] = value
        human_review["critical_failure"] = bool(
            review.get("critical_failure", False)
        )
        human_review["overall_notes"] = str(review.get("overall_notes", ""))


def refresh_objective_checks(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    cases_by_id = {case["id"]: case for case in cases}
    for record in records:
        case = cases_by_id.get(record.get("id"))
        if (
            case is None
            or record.get("status") != "completed"
            or not isinstance(record.get("output"), dict)
        ):
            continue
        checks = evaluate_checks(
            case,
            record["output"],
            int(record.get("generation_calls", 0)),
        )
        record["checks"] = checks
        record["objective_passed"] = objective_passed(checks)


def diagnostic_outputs(*generators: CountedGenerator) -> list[str]:
    diagnostics: list[str] = []
    for generator in generators:
        for output in generator.outputs:
            cleaned = re.sub(
                r"<\|channel>thought.*?<channel\|>",
                "",
                output,
                flags=re.DOTALL,
            )
            if "<|channel>thought" in cleaned:
                cleaned = cleaned.split("<|channel>thought", 1)[0]
            cleaned = cleaned.strip()
            diagnostics.append(
                cleaned[-2_000:] if cleaned else "(no approved output)"
            )
    return diagnostics


def build_payload(
    model_path: str,
    case_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record["id"])
    return {
        "metadata": {
            "model_path": model_path,
            "case_catalog": str(case_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "score_scale": {
                "1": "poor",
                "2": "material problems",
                "3": "acceptable",
                "4": "strong",
                "5": "excellent",
            },
        },
        "summary": summarize_records(ordered),
        "records": ordered,
    }


def write_reports(
    output_path: Path,
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_markdown_report(payload),
        encoding="utf-8",
    )


def run_case(
    case: dict[str, Any],
    *,
    deep: CountedGenerator,
    fast: CountedGenerator,
    background: CountedGenerator,
    background_fast: CountedGenerator | None = None,
    with_minilm: bool,
) -> dict[str, Any]:
    from structured_generation import (
        AquinasGenerationService,
        ContextualDefinition,
        ConversationGenerationMode,
        ConversationMessage,
        ConversationPersonality,
        DailyQuestionInsight,
        resolve_conversation_generation_mode,
    )

    service = AquinasGenerationService(
        deep,
        fast_generator=fast,
        background_generator=background,
        background_fast_generator=background_fast,
    )
    item = case["input"]
    operation = case["operation"]
    messages = lambda values: [
        ConversationMessage(role=value["role"], text=value["text"])
        for value in values
    ]
    concept = lambda value: ContextualDefinition(
        title=value["title"],
        part_of_speech="",
        pronunciation="",
        definition=value["definition"],
        example=value.get("example", ""),
        context=value.get("context", ""),
    )

    if operation == "conversation":
        recent_messages = messages(item["recent_messages"])
        requested_mode = ConversationGenerationMode(
            item.get("generation_mode", "automatic")
        )
        result = service.respond(
            recent_messages,
            compacted_context=item.get("compacted_context"),
            thinking_enabled=item.get("thinking_enabled", True),
            generation_mode=requested_mode,
            personality=ConversationPersonality(
                item.get("personality", "balanced")
            ),
        )
        return {
            "resolved_mode": resolve_conversation_generation_mode(
                recent_messages,
                requested_mode,
            ).value,
            "response": result.response,
            "thinking_summary": list(result.thinking_summary),
            "key_terms": [asdict(term) for term in result.key_terms],
            "insight": asdict(result.insight) if result.insight else None,
        }
    if operation == "definition":
        result = service.define_term(
            item["term"],
            source_excerpt=item.get("source_excerpt", ""),
            recent_messages=messages(item.get("recent_messages", [])),
        )
        return asdict(result)
    if operation == "question_of_the_day":
        result = service.generate_daily_question(
            item["conversation_title"],
            messages(item["recent_messages"]),
            [
                DailyQuestionInsight(
                    title=value["title"],
                    definition=value["definition"],
                )
                for value in item.get("insights", [])
            ],
        )
        return asdict(result)
    if operation == "node_label":
        result = service.label_tree_subject(item["insight_descriptions"])
        return asdict(result)
    if operation == "tree_extraction":
        result = service.analyze_tree_update(item["question"], item["response"])
        return {
            "subject_label": result.subject_label,
            "subject_summary": result.subject_summary,
            "insight_candidate": (
                asdict(result.insight_candidate)
                if result.insight_candidate
                else None
            ),
        }
    if operation == "midpoint":
        concepts = [concept(value) for value in item["concepts"]]
        candidates = service.blend_concept_candidates(concepts, item["weights"])
        output: dict[str, Any] = {
            "candidates": [asdict(candidate) for candidate in candidates],
            "selected_candidate": None,
            "candidate_similarities": [],
        }
        if with_minilm:
            selected, similarities = select_midpoint_candidate(
                concepts,
                item["weights"],
                candidates,
            )
            output["selected_candidate"] = asdict(selected)
            output["candidate_similarities"] = similarities
        return output
    if operation == "make_node":
        result = service.generate_concept_children(concept(item["concept"]))
        return {"children": [asdict(child) for child in result]}
    if operation == "compaction":
        return {
            "summary": service.compact_context(
                messages(item.get("recent_messages", [])),
                compacted_context=item.get("compacted_context"),
            )
        }
    raise ValueError(f"Unsupported operation: {operation}.")


def select_midpoint_candidate(concepts, weights, candidates):
    import numpy as np
    from relatedness import MiniLMRelatednessProvider

    provider = MiniLMRelatednessProvider()
    source_texts = [
        f"{value.title}. {value.definition}"
        for value in concepts
    ]
    candidate_texts = [
        f"{value.title}. {value.definition}"
        for value in candidates
    ]
    source_vectors = provider.embed_many(source_texts)
    numeric_weights = np.asarray(weights, dtype=np.float32)
    numeric_weights = np.clip(numeric_weights, 0, None)
    numeric_weights = numeric_weights / numeric_weights.sum()
    centroid = np.sum(
        source_vectors * numeric_weights[:, np.newaxis],
        axis=0,
    )
    centroid = centroid / np.linalg.norm(centroid)
    candidate_vectors = provider.embed_many(candidate_texts)
    similarities = candidate_vectors @ centroid
    best_index = int(np.argmax(similarities))
    return candidates[best_index], [
        {
            "title": candidate.title,
            "similarity": float(similarity),
        }
        for candidate, similarity in zip(candidates, similarities)
    ]


if __name__ == "__main__":
    main()
