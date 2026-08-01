import json
from pathlib import Path
import tempfile
import unittest

from prompt_quality import (
    SUPPORTED_OPERATIONS,
    PromptQualityConfigurationError,
    create_human_review,
    evaluate_checks,
    load_cases,
    objective_passed,
    render_markdown_report,
    resolve_path,
    summarize_records,
    validate_cases,
)
from scripts.evaluate_prompt_quality import (
    CountedGenerator,
    FirstOutputProbe,
    apply_human_reviews,
    passing_record_ids,
    refresh_objective_checks,
    run_case,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_CATALOG = REPOSITORY_ROOT / "evaluation" / "prompt_quality_cases.json"


class PromptQualityTests(unittest.TestCase):
    def test_catalog_covers_every_operation_and_repair_family(self) -> None:
        cases = load_cases(CASE_CATALOG)
        operations = {case["operation"] for case in cases}
        repair_operations = {
            case["operation"]
            for case in cases
            if "repair_probe_output" in case
        }

        self.assertEqual(operations, SUPPORTED_OPERATIONS)
        self.assertEqual(repair_operations, SUPPORTED_OPERATIONS)
        self.assertGreaterEqual(len(cases), 24)

    def test_duplicate_case_ids_are_rejected(self) -> None:
        case = {
            "id": "duplicate",
            "operation": "conversation",
            "input": {},
            "checks": [],
            "rubrics": [{"id": "quality", "criterion": "Useful."}],
        }
        with self.assertRaises(PromptQualityConfigurationError):
            validate_cases([case, case])

    def test_wildcard_paths_and_objective_checks(self) -> None:
        output = {
            "children": [
                {"title": "First", "definition": "One."},
                {"title": "Second", "definition": "Two."},
                {"title": "Third", "definition": "Three."},
            ]
        }
        case = {
            "input": {},
            "checks": [
                {"type": "count", "path": "children", "value": 3},
                {"type": "all_unique", "path": "children.*.title"},
                {"type": "nonempty", "path": "children.*.definition"},
                {"type": "generation_calls", "value": 1},
            ],
        }

        self.assertEqual(
            resolve_path(output, "children.*.title"),
            ["First", "Second", "Third"],
        )
        checks = evaluate_checks(case, output, generation_calls=1)
        self.assertTrue(objective_passed(checks))

    def test_exact_evidence_check_uses_case_input(self) -> None:
        case = {
            "input": {"response": "The exact evidence appears here."},
            "checks": [
                {
                    "type": "exact_substring_of_input",
                    "path": "candidate.evidence",
                    "input_path": "response",
                }
            ],
        }
        checks = evaluate_checks(
            case,
            {"candidate": {"evidence": "exact evidence"}},
            generation_calls=1,
        )

        self.assertTrue(checks[0]["passed"])

    def test_report_preserves_outputs_and_blank_human_rubrics(self) -> None:
        case = {
            "rubrics": [
                {"id": "accuracy", "criterion": "The answer is accurate."}
            ]
        }
        record = {
            "id": "case-1",
            "operation": "conversation",
            "description": "A test case.",
            "status": "completed",
            "duration_seconds": 1.25,
            "generation_calls": 1,
            "checks": [
                {
                    "type": "nonempty",
                    "path": "response",
                    "passed": True,
                    "expected": "nonempty",
                    "observed": "Answer",
                }
            ],
            "objective_passed": True,
            "output": {"response": "Answer"},
            "human_review": create_human_review(case),
        }
        payload = {
            "metadata": {
                "model_path": "models/Aquinas-Final",
                "generated_at": "2026-01-01T00:00:00+00:00",
            },
            "summary": summarize_records([record]),
            "records": [record],
        }

        report = render_markdown_report(payload)

        self.assertIn("Objective pass rate: **100%**", report)
        self.assertIn('"response": "Answer"', report)
        self.assertIn("accuracy — —/5", report)

    def test_suite_pass_requires_complete_passing_human_review(self) -> None:
        record = {
            "status": "completed",
            "objective_passed": True,
            "human_review": {
                "scores": [{"id": "quality", "score": 4}],
                "critical_failure": False,
            },
        }

        summary = summarize_records([record])

        self.assertTrue(summary["human_review_complete"])
        self.assertTrue(summary["suite_passed"])

        record["human_review"]["scores"][0]["score"] = 2
        self.assertFalse(summarize_records([record])["suite_passed"])

        record["human_review"]["scores"][0]["score"] = 4
        record["human_review"]["critical_failure"] = True
        self.assertFalse(summarize_records([record])["suite_passed"])

    def test_load_cases_requires_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"id": "not-an-array"}))
            with self.assertRaises(PromptQualityConfigurationError):
                load_cases(path)

    def test_runner_dispatches_definition_and_records_repair_calls(self) -> None:
        generated = iter(
            [
                '{"title":"Prudence","context":"Concrete action",'
                '"definition":"Right reason about things to be done."}'
            ]
        )
        deep = CountedGenerator(
            lambda _prompt, _limit: next(generated),
            first_output="not json",
        )
        unused_fast = CountedGenerator(lambda _prompt, _limit: "")
        unused_background = CountedGenerator(lambda _prompt, _limit: "")
        case = {
            "operation": "definition",
            "input": {
                "term": "prudence",
                "source_excerpt": "Prudence directs action.",
                "recent_messages": [],
            },
        }

        output = run_case(
            case,
            deep=deep,
            fast=unused_fast,
            background=unused_background,
            with_minilm=False,
        )

        self.assertEqual(output["title"], "Prudence")
        self.assertEqual(output["part_of_speech"], "")
        self.assertEqual(deep.calls, 2)
        self.assertEqual(unused_fast.calls, 0)
        self.assertEqual(unused_background.calls, 0)

    def test_resume_skips_only_completed_objective_passes(self) -> None:
        records = [
            {"id": "pass", "status": "completed", "objective_passed": True},
            {"id": "fail", "status": "completed", "objective_passed": False},
            {"id": "error", "status": "error", "objective_passed": False},
        ]

        self.assertEqual(passing_record_ids(records), {"pass"})

    def test_versioned_review_catalog_applies_by_case_and_rubric_id(self) -> None:
        records = [
            {
                "id": "case-1",
                "human_review": {
                    "scores": [
                        {"id": "accuracy", "score": None},
                        {"id": "restraint", "score": None},
                    ],
                    "critical_failure": None,
                    "overall_notes": "",
                },
            }
        ]

        apply_human_reviews(
            records,
            {
                "case-1": {
                    "scores": {"accuracy": 5, "restraint": 4},
                    "critical_failure": False,
                    "overall_notes": "Reviewed against the saved output.",
                }
            },
        )

        review = records[0]["human_review"]
        self.assertEqual(
            [score["score"] for score in review["scores"]],
            [5, 4],
        )
        self.assertFalse(review["critical_failure"])
        self.assertIn("saved output", review["overall_notes"])

    def test_render_refreshes_objective_checks_from_current_catalog(self) -> None:
        records = [
            {
                "id": "case-1",
                "status": "completed",
                "generation_calls": 1,
                "output": {"response": "A grounded answer."},
                "checks": [],
                "objective_passed": False,
            }
        ]
        cases = [
            {
                "id": "case-1",
                "input": {},
                "checks": [
                    {"type": "nonempty", "path": "response"},
                    {"type": "generation_calls", "value": 1},
                ],
            }
        ]

        refresh_objective_checks(records, cases)

        self.assertTrue(records[0]["objective_passed"])
        self.assertEqual(len(records[0]["checks"]), 2)

    def test_repair_probe_is_consumed_once_across_generator_paths(self) -> None:
        probe = FirstOutputProbe("invalid")
        deep = CountedGenerator(lambda _prompt, _limit: "deep", probe=probe)
        background_fast = CountedGenerator(
            lambda _prompt, _limit: "background-fast",
            probe=probe,
        )

        self.assertEqual(deep("prompt", 10), "invalid")
        self.assertEqual(background_fast("repair", 10), "background-fast")


if __name__ == "__main__":
    unittest.main()
