import json
import unittest

from structured_generation import (
    AquinasGenerationService,
    COMPACTION_GENERATION_MAX_TOKENS,
    CONVERSATION_GENERATION_MAX_TOKENS,
    ConversationMessage,
    ConversationStreamParser,
    STRUCTURED_GENERATION_MAX_TOKENS,
    TREE_ANALYSIS_MAX_TOKENS,
    StructuredGenerationError,
)


class RecordingGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, int]] = []

    def __call__(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        return self.outputs.pop(0)


class AquinasGenerationServiceTests(unittest.TestCase):
    def test_extracts_only_the_central_grounded_tree_seed(self) -> None:
        response = (
            "Natural law is the rational creature's participation in eternal law. "
            "Its first principle is that good is to be done and pursued."
        )
        generator = RecordingGenerator(
            [
                """
                {
                  "subject": {
                    "label": "Natural Law",
                    "summary": "Reason participates in eternal law."
                  },
                  "node_seed": {
                    "label": "Participation in Eternal Law",
                    "summary": "Natural law is rational participation in eternal law.",
                    "evidence_excerpt": "Natural law is the rational creature's participation in eternal law."
                  }
                }
                """
            ]
        )

        result = AquinasGenerationService(generator).analyze_tree_update(
            "What is natural law?",
            response,
        )

        self.assertEqual(result.subject_label, "Natural Law")
        self.assertEqual(result.node_seed.label, "Participation in Eternal Law")
        self.assertIn("<TASK:INSIGHT_TREE_UPDATE>", generator.calls[0][0])
        self.assertEqual(generator.calls[0][1], TREE_ANALYSIS_MAX_TOKENS)

    def test_tree_analysis_uses_visible_text_from_aq_links(self) -> None:
        annotated_response = (
            "[Natural law](aq://natural-law) is participation in eternal law."
        )
        generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "subject": {
                            "label": "Natural Law",
                            "summary": "Reason participates in eternal law.",
                        },
                        "node_seed": {
                            "label": "Participation in Eternal Law",
                            "summary": "Natural law participates in eternal law.",
                            "evidence_excerpt": (
                                "Natural law is participation in eternal law."
                            ),
                        },
                    }
                )
            ]
        )

        result = AquinasGenerationService(generator).analyze_tree_update(
            "What is natural law?",
            annotated_response,
        )

        self.assertIsNotNone(result.node_seed)
        self.assertNotIn("aq://", generator.calls[0][0])
        self.assertIn(
            "Highlighted terms in the answer:\nNatural law",
            generator.calls[0][0],
        )

    def test_generates_node_subject_distinct_from_insight_title(self) -> None:
        generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "label": "Divine Providence",
                        "summary": "God's wisdom orders created causes toward their ends.",
                    }
                )
            ]
        )

        subject = AquinasGenerationService(generator).label_tree_subject(
            ["Divine Will: God's willing is identical with the divine essence."]
        )

        self.assertEqual(subject.label, "Divine Providence")
        self.assertIn("<TASK:INSIGHT_TREE_NODE_SUBJECT>", generator.calls[0][0])

    def test_discards_ungrounded_tree_node_seed(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "subject": {"label": "Grace", "summary": "A summary."},
                  "node_seed": {
                    "label": "Created Grace",
                    "summary": "A durable definition.",
                    "evidence_excerpt": "This never appeared."
                  }
                }
                """
            ]
        )

        result = AquinasGenerationService(generator).analyze_tree_update(
            "Thanks?",
            "You are welcome.",
        )

        self.assertIsNone(result.node_seed)

    def test_discards_acknowledgement_and_procedural_tree_candidates(self) -> None:
        outputs = iter(
            [
                json.dumps(
                    {
                        "subject": {"label": "Courtesy", "summary": "A reply."},
                        "node_seed": {
                            "label": "Polite Acknowledgement",
                            "summary": "The assistant acknowledges gratitude.",
                            "evidence_excerpt": "You're welcome!",
                        },
                    }
                ),
                json.dumps(
                    {
                        "subject": {"label": "Interface Steps", "summary": "A step."},
                        "node_seed": {
                            "label": "Save Button Action",
                            "summary": "The next action uses the Save control.",
                            "evidence_excerpt": "Click Save to continue.",
                        },
                    }
                ),
            ]
        )
        service = AquinasGenerationService(generator=lambda _prompt, _limit: next(outputs))

        acknowledgement = service.analyze_tree_update("Thanks?", "You're welcome!")
        procedural = service.analyze_tree_update(
            "What now?",
            "Click Save to continue.",
        )

        self.assertIsNone(acknowledgement.node_seed)
        self.assertIsNone(procedural.node_seed)

    def test_compacts_existing_checkpoint_and_recent_turns(self) -> None:
        generator = RecordingGenerator(
            ['{"summary":"The user is studying natural law and has distinguished it from civil law."}']
        )
        service = AquinasGenerationService(generator)

        summary = service.compact_context(
            [
                ConversationMessage(
                    role="user",
                    text="How is natural law different from civil law?",
                )
            ],
            compacted_context="The discussion concerns kinds of law.",
        )

        self.assertIn("natural law", summary)
        self.assertIn(
            "The discussion concerns kinds of law.",
            generator.calls[0][0],
        )
        self.assertEqual(
            generator.calls[0][1],
            COMPACTION_GENERATION_MAX_TOKENS,
        )

    def test_conversation_prompt_includes_compacted_context(self) -> None:
        generator = RecordingGenerator(
            ['{"response":"Answer.","thinking_summary":[],"key_terms":[]}']
        )
        service = AquinasGenerationService(generator)

        service.respond(
            [ConversationMessage(role="user", text="Continue.")],
            compacted_context="Earlier conclusions.",
        )

        self.assertIn("Earlier conclusions.", generator.calls[0][0])

    def test_stream_parser_exposes_summary_then_response_deltas(self) -> None:
        parser = ConversationStreamParser()

        updates = []
        for fragment in [
            '<|channel>thought private scratch work<channel|>{"thinking_',
            'summary":["Compared the two ideas.","Applied the distinction"],',
            '"response":"Grace perfects ',
            'nature.\\nIt does not erase it.","key_terms":[]}',
        ]:
            updates.extend(parser.feed(fragment))

        summaries = [
            update.value
            for update in updates
            if update.kind == "thinking_summary"
        ]
        deltas = [
            update.value
            for update in updates
            if update.kind == "response_delta"
        ]

        self.assertEqual(
            summaries[-1],
            ("Compared the two ideas.", "Applied the distinction"),
        )
        self.assertEqual(
            "".join(deltas),
            "Grace perfects nature.\nIt does not erase it.",
        )

    def test_stream_parser_waits_for_complete_json_escape(self) -> None:
        parser = ConversationStreamParser()

        first = parser.feed('{"thinking_summary":[],"response":"Line one\\')
        second = parser.feed('nLine two","key_terms":[]}')

        self.assertEqual(
            "".join(
                update.value
                for update in first + second
                if update.kind == "response_delta"
            ),
            "Line one\nLine two",
        )

    def test_parses_conversation_response_and_validates_terms(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "response": "Natural law participates in eternal law through human reason.",
                  "thinking_summary": [
                    "Distinguished natural law from eternal law.",
                    "Connected both through human reason."
                  ],
                  "key_terms": [
                    {
                      "display_text": "natural LAW",
                      "canonical_term": "Natural Law",
                      "context_excerpt": "wrong excerpt"
                    },
                    {
                      "display_text": "term not in answer",
                      "canonical_term": "Missing",
                      "context_excerpt": ""
                    }
                  ]
                }
                """
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.respond(
            [ConversationMessage(role="user", text="What is natural law?")],
            thinking_enabled=True,
        )

        self.assertEqual(
            result.response,
            "Natural law participates in eternal law through human reason.",
        )
        self.assertEqual(len(result.key_terms), 1)
        self.assertEqual(
            result.thinking_summary,
            (
                "Distinguished natural law from eternal law.",
                "Connected both through human reason.",
            ),
        )
        self.assertEqual(result.key_terms[0].display_text, "Natural law")
        self.assertIn(
            result.key_terms[0].display_text,
            result.key_terms[0].context_excerpt,
        )
        self.assertEqual(
            generator.calls[0][1],
            CONVERSATION_GENERATION_MAX_TOKENS,
        )

    def test_retries_invalid_conversation_response_once(self) -> None:
        generator = RecordingGenerator(
            [
                "A prose answer without JSON.",
                """
                {
                  "response": "Grace perfects nature.",
                  "thinking_summary": [
                    "Related grace to the perfection of human nature."
                  ],
                  "key_terms": []
                }
                """,
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.respond(
            [ConversationMessage(role="user", text="How does grace relate to nature?")],
            thinking_enabled=True,
        )

        self.assertEqual(result.response, "Grace perfects nature.")
        self.assertEqual(len(generator.calls), 2)
        self.assertIn(
            "<TASK:REPAIR_CONVERSATION_JSON>",
            generator.calls[1][0],
        )

    def test_thinking_is_disabled_by_default(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "response": "Truth perfects the intellect.",
                  "thinking_summary": [
                    "This should be suppressed when Thinking is off."
                  ],
                  "key_terms": []
                }
                """
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.respond(
            [ConversationMessage(role="user", text="What does truth do?")]
        )

        self.assertEqual(result.thinking_summary, ())
        self.assertIn(
            "Thinking mode is disabled",
            generator.calls[0][0],
        )
        self.assertIn(
            "Return an empty\nthinking_summary array",
            generator.calls[0][0],
        )

    def test_parses_contextual_definition_json(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "title": "Natural Law",
                  "part_of_speech": "noun",
                  "pronunciation": "",
                  "definition": "Natural law is reason's participation in eternal law.",
                  "example": "A person recognizes that justice should guide action."
                }
                """
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.define_term(
            term="Natural Law",
            source_excerpt="The precepts of natural law...",
            recent_messages=[
                ConversationMessage(
                    role="user",
                    text="How can moral truth be known?",
                )
            ],
        )

        self.assertEqual(result.title, "Natural Law")
        self.assertIn("eternal law", result.definition)
        self.assertEqual(len(generator.calls), 1)
        self.assertIn("<TASK:CONTEXTUAL_DEFINITION>", generator.calls[0][0])
        self.assertEqual(
            generator.calls[0][1],
            STRUCTURED_GENERATION_MAX_TOKENS,
        )

    def test_extracts_json_from_extra_model_text(self) -> None:
        generator = RecordingGenerator(
            [
                """
                Here is the requested result:
                ```json
                {
                  "word": "Essence",
                  "meaning": "Essence identifies what a thing is."
                }
                ```
                """
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.define_term(term="Essence")

        self.assertEqual(result.title, "Essence")
        self.assertEqual(
            result.definition,
            "Essence identifies what a thing is.",
        )

    def test_retries_once_when_first_output_is_invalid(self) -> None:
        generator = RecordingGenerator(
            [
                "Essence is an important metaphysical concept.",
                """
                {
                  "title": "Essence",
                  "part_of_speech": "noun",
                  "pronunciation": "",
                  "definition": "Essence describes what a thing is.",
                  "example": ""
                }
                """,
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.define_term(term="Essence")

        self.assertEqual(result.title, "Essence")
        self.assertEqual(len(generator.calls), 2)
        self.assertIn("<TASK:REPAIR_JSON>", generator.calls[1][0])

    def test_fails_after_one_invalid_retry(self) -> None:
        generator = RecordingGenerator(["not json", "still not json"])
        service = AquinasGenerationService(generator)

        with self.assertRaises(StructuredGenerationError):
            service.define_term(term="Existence")


if __name__ == "__main__":
    unittest.main()
