"""Case validation, objective checks, and reporting for prompt-quality evaluations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_OPERATIONS = {
    "conversation",
    "definition",
    "question_of_the_day",
    "node_label",
    "tree_extraction",
    "midpoint",
    "make_node",
    "compaction",
}
SUPPORTED_CHECKS = {
    "all_empty",
    "all_unique",
    "count",
    "count_at_most",
    "ends_with",
    "equals",
    "exact_substring_of_input",
    "generation_calls",
    "is_null",
    "max_words",
    "min_words",
    "nonempty",
    "not_contains_ci",
    "not_null",
}


class PromptQualityConfigurationError(ValueError):
    """Raised when the evaluation catalog is not internally consistent."""


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise PromptQualityConfigurationError("The case catalog must be a JSON array.")
    validate_cases(raw)
    return raw


def validate_cases(cases: Sequence[dict[str, Any]]) -> None:
    identifiers: list[str] = []
    operations: Counter[str] = Counter()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise PromptQualityConfigurationError(
                f"Case {index} must be a JSON object."
            )
        case_id = _required_string(case, "id", index)
        operation = _required_string(case, "operation", index)
        if operation not in SUPPORTED_OPERATIONS:
            raise PromptQualityConfigurationError(
                f"{case_id}: unsupported operation {operation!r}."
            )
        if not isinstance(case.get("input"), dict):
            raise PromptQualityConfigurationError(
                f"{case_id}: input must be an object."
            )
        checks = case.get("checks", [])
        if not isinstance(checks, list):
            raise PromptQualityConfigurationError(
                f"{case_id}: checks must be an array."
            )
        for check_index, check in enumerate(checks):
            if not isinstance(check, dict):
                raise PromptQualityConfigurationError(
                    f"{case_id}: check {check_index} must be an object."
                )
            check_type = check.get("type")
            if check_type not in SUPPORTED_CHECKS:
                raise PromptQualityConfigurationError(
                    f"{case_id}: unsupported check type {check_type!r}."
                )
            if check_type != "generation_calls" and not isinstance(
                check.get("path"),
                str,
            ):
                raise PromptQualityConfigurationError(
                    f"{case_id}: check {check_index} requires a path."
                )
        rubrics = case.get("rubrics", [])
        if not isinstance(rubrics, list) or not rubrics:
            raise PromptQualityConfigurationError(
                f"{case_id}: at least one human rubric is required."
            )
        for rubric_index, rubric in enumerate(rubrics):
            if (
                not isinstance(rubric, dict)
                or not isinstance(rubric.get("id"), str)
                or not isinstance(rubric.get("criterion"), str)
            ):
                raise PromptQualityConfigurationError(
                    f"{case_id}: rubric {rubric_index} needs id and criterion."
                )
        identifiers.append(case_id)
        operations[operation] += 1

    duplicates = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        raise PromptQualityConfigurationError(
            f"Duplicate case ids: {', '.join(duplicates)}."
        )
    missing_operations = sorted(SUPPORTED_OPERATIONS - operations.keys())
    if missing_operations:
        raise PromptQualityConfigurationError(
            "The catalog has no coverage for: " + ", ".join(missing_operations)
        )


def evaluate_checks(
    case: dict[str, Any],
    output: dict[str, Any],
    generation_calls: int,
) -> list[dict[str, Any]]:
    results = []
    for check in case.get("checks", []):
        passed, observed = _evaluate_check(
            check,
            case_input=case["input"],
            output=output,
            generation_calls=generation_calls,
        )
        results.append(
            {
                "type": check["type"],
                "path": check.get("path"),
                "passed": passed,
                "expected": _expected_description(check),
                "observed": observed,
            }
        )
    return results


def objective_passed(checks: Sequence[dict[str, Any]]) -> bool:
    return bool(checks) and all(check["passed"] for check in checks)


def create_human_review(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "scores": [
            {
                "id": rubric["id"],
                "criterion": rubric["criterion"],
                "score": None,
                "notes": "",
            }
            for rubric in case["rubrics"]
        ],
        "critical_failure": None,
        "overall_notes": "",
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    objective_passes = sum(
        record.get("objective_passed") is True
        for record in completed
    )
    failures = sum(record.get("status") == "error" for record in records)
    reviewed_scores = [
        score["score"]
        for record in records
        for score in record.get("human_review", {}).get("scores", [])
        if isinstance(score.get("score"), (int, float))
    ]
    invalid_scores = [
        score
        for score in reviewed_scores
        if not 1 <= score <= 5
    ]
    if invalid_scores:
        raise PromptQualityConfigurationError(
            "Human rubric scores must be between 1 and 5."
        )
    human_review_complete = bool(records) and all(
        score.get("score") is not None
        for record in records
        for score in record.get("human_review", {}).get("scores", [])
    )
    critical_failures = sum(
        record.get("human_review", {}).get("critical_failure") is True
        for record in records
    )
    human_scores_pass = bool(reviewed_scores) and all(
        score >= 3
        for score in reviewed_scores
    )
    objective_suite_pass = (
        len(completed) == len(records)
        and objective_passes == len(records)
    )
    return {
        "cases": len(records),
        "completed": len(completed),
        "generation_errors": failures,
        "objective_passes": objective_passes,
        "objective_pass_rate": (
            objective_passes / len(completed)
            if completed
            else None
        ),
        "human_review_complete": human_review_complete,
        "mean_human_score": (
            sum(reviewed_scores) / len(reviewed_scores)
            if reviewed_scores
            else None
        ),
        "critical_failures": critical_failures,
        "suite_passed": (
            objective_suite_pass
            and human_review_complete
            and human_scores_pass
            and critical_failures == 0
        ),
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    records = payload.get("records", [])
    summary = payload.get("summary") or summarize_records(records)
    metadata = payload.get("metadata", {})
    lines = [
        "# Aquinas Prompt Quality Report",
        "",
        f"- Model: `{metadata.get('model_path', 'unknown')}`",
        f"- Generated: `{metadata.get('generated_at', 'unknown')}`",
        f"- Cases: **{summary['cases']}**",
        f"- Completed: **{summary['completed']}**",
        f"- Objective pass rate: **{_format_rate(summary['objective_pass_rate'])}**",
        f"- Generation errors: **{summary['generation_errors']}**",
        f"- Human review complete: **{summary['human_review_complete']}**",
        f"- Mean human score: **{_format_score(summary['mean_human_score'])}**",
        f"- Critical failures: **{summary['critical_failures']}**",
        f"- Suite passed: **{summary['suite_passed']}**",
        "",
        "## Summary",
        "",
        "| Case | Operation | Objective | Human review | Duration |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        objective = (
            "PASS"
            if record.get("objective_passed") is True
            else "FAIL"
            if record.get("status") == "completed"
            else "ERROR"
        )
        review = _human_review_summary(record.get("human_review", {}))
        duration = record.get("duration_seconds")
        lines.append(
            "| {id} | {operation} | {objective} | {review} | {duration} |".format(
                id=record.get("id", ""),
                operation=record.get("operation", ""),
                objective=objective,
                review=review,
                duration=f"{duration:.2f}s" if isinstance(duration, float) else "—",
            )
        )

    lines.extend(["", "## Case details", ""])
    for record in records:
        lines.extend(
            [
                f"### {record.get('id', '')}",
                "",
                record.get("description", ""),
                "",
                f"- Operation: `{record.get('operation', '')}`",
                f"- Status: `{record.get('status', '')}`",
                f"- Model calls: `{record.get('generation_calls', 0)}`",
                "",
                "Objective checks:",
                "",
            ]
        )
        for check in record.get("checks", []):
            mark = "x" if check.get("passed") else " "
            path = f" at `{check['path']}`" if check.get("path") else ""
            lines.append(
                f"- [{mark}] `{check.get('type')}`{path}: "
                f"expected `{_short_json(check.get('expected'))}`; observed "
                f"`{_short_json(check.get('observed'))}`"
            )
        if record.get("error"):
            lines.extend(["", f"Error: `{record['error']}`"])
        if record.get("diagnostic_outputs"):
            lines.extend(
                [
                    "",
                    "<details>",
                    "<summary>Sanitized diagnostic output</summary>",
                    "",
                ]
            )
            for index, output in enumerate(record["diagnostic_outputs"], start=1):
                lines.extend(
                    [
                        f"Generation {index}:",
                        "",
                        "```text",
                        output,
                        "```",
                        "",
                    ]
                )
            lines.append("</details>")
        if record.get("output") is not None:
            lines.extend(
                [
                    "",
                    "<details>",
                    "<summary>Generated output</summary>",
                    "",
                    "```json",
                    json.dumps(record["output"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "</details>",
                ]
            )
        lines.extend(["", "Human review (1 = poor, 5 = excellent):", ""])
        for score in record.get("human_review", {}).get("scores", []):
            value = "—" if score.get("score") is None else str(score["score"])
            lines.append(
                f"- **{score['id']} — {value}/5:** {score['criterion']}  "
            )
            if score.get("notes"):
                lines.append(f"  Notes: {score['notes']}")
        lines.extend(
            [
                "",
                f"- Critical failure: "
                f"`{record.get('human_review', {}).get('critical_failure')}`",
                f"- Overall notes: "
                f"{record.get('human_review', {}).get('overall_notes', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _evaluate_check(
    check: dict[str, Any],
    *,
    case_input: dict[str, Any],
    output: dict[str, Any],
    generation_calls: int,
) -> tuple[bool, Any]:
    check_type = check["type"]
    if check_type == "generation_calls":
        expected = int(check["value"])
        return generation_calls == expected, generation_calls

    observed = resolve_path(output, check["path"])
    values = observed if isinstance(observed, list) else [observed]
    if check_type == "nonempty":
        return all(_is_nonempty(value) for value in values), observed
    if check_type == "all_empty":
        return all(not _is_nonempty(value) for value in values), observed
    if check_type == "is_null":
        return observed is None, observed
    if check_type == "not_null":
        return observed is not None, observed
    if check_type == "equals":
        return observed == check.get("value"), observed
    if check_type == "count":
        return _safe_length(observed) == int(check["value"]), _safe_length(observed)
    if check_type == "count_at_most":
        length = _safe_length(observed)
        return length is not None and length <= int(check["value"]), length
    if check_type == "max_words":
        counts = [_word_count(value) for value in values]
        return all(count <= int(check["value"]) for count in counts), counts
    if check_type == "min_words":
        counts = [_word_count(value) for value in values]
        return all(count >= int(check["value"]) for count in counts), counts
    if check_type == "ends_with":
        suffix = str(check["value"])
        return all(isinstance(value, str) and value.endswith(suffix) for value in values), observed
    if check_type == "all_unique":
        normalized = [str(value).strip().casefold() for value in values]
        return len(normalized) == len(set(normalized)), observed
    if check_type == "not_contains_ci":
        haystack = json.dumps(observed, ensure_ascii=False).casefold()
        forbidden = [str(value) for value in check.get("values", [])]
        matches = [value for value in forbidden if value.casefold() in haystack]
        return not matches, matches
    if check_type == "exact_substring_of_input":
        source = resolve_path(case_input, str(check["input_path"]))
        return (
            isinstance(observed, str)
            and isinstance(source, str)
            and observed in source
        ), observed
    raise PromptQualityConfigurationError(f"Unsupported check type: {check_type}.")


def resolve_path(value: Any, path: str) -> Any:
    if path in {"", "$"}:
        return value
    current: Any = value
    for component in path.split("."):
        if component == "*":
            if not isinstance(current, list):
                return []
            remaining = ".".join(path.split(".")[path.split(".").index(component) + 1:])
            return [
                nested
                for item in current
                for nested in _as_list(resolve_path(item, remaining))
            ]
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _required_string(case: dict[str, Any], key: str, index: int) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PromptQualityConfigurationError(
            f"Case {index} requires a non-empty {key}."
        )
    return value


def _expected_description(check: dict[str, Any]) -> Any:
    if "value" in check:
        return check["value"]
    if "values" in check:
        return {"none_of": check["values"]}
    if "input_path" in check:
        return {"exact_substring_of_input": check["input_path"]}
    return check["type"]


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _is_nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _safe_length(value: Any) -> int | None:
    return len(value) if isinstance(value, (str, list, dict, tuple)) else None


def _word_count(value: Any) -> int:
    return len(str(value).split()) if isinstance(value, str) else 0


def _format_rate(value: Any) -> str:
    return f"{value:.0%}" if isinstance(value, float) else "—"


def _format_score(value: Any) -> str:
    return f"{value:.2f}/5" if isinstance(value, float) else "—"


def _short_json(value: Any, limit: int = 120) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _human_review_summary(review: dict[str, Any]) -> str:
    scores = [
        score.get("score")
        for score in review.get("scores", [])
        if isinstance(score.get("score"), (int, float))
    ]
    return f"{sum(scores) / len(scores):.1f}/5" if scores else "Pending"
