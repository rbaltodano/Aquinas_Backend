"""Run the reviewed latency/quality set against one Aquinas checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("evaluation/latency_quality_prompts.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("automatic", "fast", "deep"),
        default="automatic",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases for a smoke benchmark.",
    )
    args = parser.parse_args()

    os.environ["AQUINAS_MODEL_PATH"] = args.model_path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from main import generate_aquinas_stream
    from structured_generation import (
        AquinasGenerationService,
        ConversationGenerationMode,
        ConversationMessage,
        ConversationStreamParser,
        resolve_conversation_generation_mode,
    )

    prompts = json.loads(args.prompts.read_text())
    if args.limit is not None:
        prompts = prompts[:max(0, args.limit)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    service = AquinasGenerationService(lambda _prompt, _limit: "")
    requested_mode = ConversationGenerationMode(args.mode)
    records: list[dict] = []

    for item in prompts:
        messages = [ConversationMessage(role="user", text=item["question"])]
        resolved_mode = resolve_conversation_generation_mode(
            messages,
            requested_mode,
        )
        prompt = service.conversation_prompt(
            messages,
            thinking_enabled=True,
            generation_mode=resolved_mode,
        )
        prefix = "{" if resolved_mode == ConversationGenerationMode.FAST else ""
        parser_state = ConversationStreamParser()
        if prefix:
            parser_state.feed(prefix)
        raw_output = [prefix] if prefix else []
        started_at = time.perf_counter()
        first_token_at = None
        first_approved_at = None
        first_response_at = None
        final_fragment = None

        for fragment in generate_aquinas_stream(
            prompt,
            max_tokens=1_800,
            response_prefix=prefix,
        ):
            now = time.perf_counter()
            final_fragment = now
            if first_token_at is None:
                first_token_at = now
            raw_output.append(fragment)
            for update in parser_state.feed(fragment):
                if first_approved_at is None:
                    first_approved_at = now
                if update.kind == "response_delta" and first_response_at is None:
                    first_response_at = now

        completed_at = final_fragment or time.perf_counter()
        output = "".join(raw_output)
        valid = True
        response = None
        try:
            parsed = service.parse_conversation_output(output)
            response = {
                "text": parsed.response,
                "thinking_summary": list(parsed.thinking_summary),
                "key_terms": [
                    {
                        "display_text": term.display_text,
                        "canonical_term": term.canonical_term,
                        "context_excerpt": term.context_excerpt,
                    }
                    for term in parsed.key_terms
                ],
            }
        except Exception:
            valid = False

        record = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "requested_mode": requested_mode.value,
            "resolved_mode": resolved_mode.value,
            "valid": valid,
            "ttft_seconds": (
                first_token_at - started_at if first_token_at is not None else None
            ),
            "first_approved_field_seconds": (
                first_approved_at - started_at
                if first_approved_at is not None
                else None
            ),
            "first_response_text_seconds": (
                first_response_at - started_at
                if first_response_at is not None
                else None
            ),
            "total_seconds": completed_at - started_at,
            "response": response,
            "human_review": {
                "no_worse_than_bf16": None,
                "critical_regression": None,
                "notes": "",
            },
        }
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "response"}))

    args.output.write_text(json.dumps(records, indent=2) + "\n")
    totals = [record["total_seconds"] for record in records]
    first_responses = [
        record["first_response_text_seconds"]
        for record in records
        if record["first_response_text_seconds"] is not None
    ]
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "cases": len(records),
                "valid_cases": sum(record["valid"] for record in records),
                "p50_total_seconds": statistics.median(totals),
                "p50_first_response_seconds": (
                    statistics.median(first_responses)
                    if first_responses
                    else None
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
