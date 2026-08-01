import json
import unittest

from structured_generation import (
    AquinasGenerationService,
    COMPACTION_GENERATION_MAX_TOKENS,
    CONVERSATION_GENERATION_MAX_TOKENS,
    DAILY_QUESTION_MAX_TOKENS,
    MIDPOINT_GENERATION_MAX_TOKENS,
    MAKE_NODE_GENERATION_MAX_TOKENS,
    ConversationGenerationMode,
    ConversationImage,
    ConversationPersonality,
    ConversationMessage,
    ConversationStreamParser,
    ContextualDefinition,
    DailyQuestionInsight,
    STRUCTURED_GENERATION_MAX_TOKENS,
    TREE_ANALYSIS_MAX_TOKENS,
    StructuredGenerationError,
    requested_definition_term,
    resolve_conversation_generation_mode,
)


class RecordingGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, int]] = []
        self.image_calls: list[tuple[bytes, ...]] = []

    def __call__(
        self,
        prompt: str,
        max_tokens: int,
        *,
        images: tuple[bytes, ...] = (),
    ) -> str:
        self.calls.append((prompt, max_tokens))
        self.image_calls.append(images)
        return self.outputs.pop(0)


class AquinasGenerationServiceTests(unittest.TestCase):
    def test_structured_application_prompts_require_persona_independent_language(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))
        concept = ContextualDefinition(
            title="Prudence",
            part_of_speech="noun",
            pronunciation="",
            definition="Practical wisdom applied to action.",
            example="Choosing a fitting means toward a good end.",
        )
        neutral_prompts = (
            service._definition_prompt(
                "prudence",
                "Prudence guides action.",
                [ConversationMessage(role="user", text="What is prudence?")],
            ),
            service._daily_question_prompt(
                "Practical Wisdom",
                [ConversationMessage(role="user", text="How should I choose?")],
                [
                    DailyQuestionInsight(
                        title="Prudence",
                        definition="Practical wisdom applied to action.",
                    )
                ],
            ),
            service._compaction_prompt(
                [ConversationMessage(role="user", text="How should I choose?")],
                None,
            ),
            service._tree_analysis_prompt(
                "What is prudence?",
                "Prudence guides action.",
            ),
            service._node_subject_prompt(["Prudence. Practical wisdom."]),
            service._concept_blend_prompt((concept, concept), (0.5, 0.5)),
            service._concept_children_prompt(concept),
        )

        for prompt in neutral_prompts:
            with self.subTest(task=prompt.splitlines()[0]):
                self.assertIn(
                    "independent of any conversational persona",
                    prompt,
                )
        definition_prompt = neutral_prompts[0]
        self.assertIn("exact relational object and direction", definition_prompt)
        self.assertIn("received or possessed in a limited way", definition_prompt)
        node_prompt = neutral_prompts[4]
        self.assertIn("cover the whole Insight", node_prompt)
        self.assertIn("mixed types", node_prompt)
        midpoint_prompt = neutral_prompts[5]
        self.assertIn("one integrated weighted center", midpoint_prompt)
        self.assertIn("Never distribute the pool", midpoint_prompt)
        children_prompt = neutral_prompts[6]
        self.assertIn("contribute a new claim", children_prompt)
        self.assertIn("generic bucket", children_prompt)

    def test_repair_prompts_require_minimal_non_creative_correction(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))
        concept = ContextualDefinition(
            title="Prudence",
            part_of_speech="",
            pronunciation="",
            definition="Practical wisdom applied to action.",
            example="Choosing a fitting means toward a good end.",
        )
        repair_prompts = (
            service._daily_question_repair_prompt(
                "invalid",
                "Practical Wisdom",
                (ConversationMessage(role="user", text="How should I choose?"),),
                (
                    DailyQuestionInsight(
                        title="Prudence",
                        definition="Practical wisdom applied to action.",
                    ),
                ),
            ),
            service._compaction_repair_prompt(
                "invalid",
                (ConversationMessage(role="user", text="What changed?"),),
                "The earlier position was tentative.",
            ),
            service._node_subject_repair_prompt(
                "invalid",
                ("Prudence. Practical wisdom applied to action.",),
            ),
            service._concept_blend_repair_prompt(
                "invalid",
                (concept, concept),
                (0.75, 0.25),
            ),
            service._concept_children_repair_prompt("invalid", concept),
            service._tree_analysis_repair_prompt(
                "invalid",
                "Visible question?",
                "Visible answer.",
            ),
            service._conversation_repair_prompt(
                "invalid",
                thinking_enabled=True,
            ),
            service._repair_prompt("prudence", "invalid"),
        )

        for prompt in repair_prompts:
            with self.subTest(task=prompt.splitlines()[0]):
                self.assertIn("only the invalid or missing contract surface", prompt)
                self.assertIn("Preserve valid substantive content", prompt)
                self.assertIn("Do not stylistically rewrite", prompt)
                self.assertIn("unsupported claims", prompt)
                self.assertIn("safest schema-permitted null or empty value", prompt)
                self.assertIn("Never expose hidden", prompt)
                self.assertIn("Use neutral, clear editorial language", prompt)
                self.assertIn("conversational\npersona", prompt)

        self.assertIn("Prefer null", repair_prompts[0])
        self.assertIn("The earlier position was tentative", repair_prompts[1])
        self.assertIn("Practical wisdom applied to action", repair_prompts[2])
        self.assertIn('"weight_percent": 75.0', repair_prompts[3])
        self.assertIn("Original Node Concept", repair_prompts[4])
        self.assertIn(
            "Remove an invalid key term rather than altering valid response prose",
            repair_prompts[6],
        )

    def test_daily_question_uses_background_generator_and_validates_citation(self) -> None:
        foreground_generator = RecordingGenerator([])
        background_generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "question": "How might prudence reshape the choice you described?",
                        "reason_for_asking": "Your conversation left the practical choice open.",
                        "cited_insight_title": "Prudence",
                    }
                )
            ]
        )
        service = AquinasGenerationService(
            foreground_generator,
            background_generator=background_generator,
        )

        result = service.generate_daily_question(
            "Practical Wisdom",
            [
                ConversationMessage(role="user", text="How should I choose well?"),
                ConversationMessage(role="assistant", text="Prudence guides concrete action."),
            ],
            [
                DailyQuestionInsight(
                    title="Prudence",
                    definition="Practical wisdom applied to action.",
                )
            ],
        )

        self.assertEqual(result.cited_insight_title, "Prudence")
        self.assertEqual(foreground_generator.calls, [])
        self.assertEqual(background_generator.calls[0][1], DAILY_QUESTION_MAX_TOKENS)
        self.assertIn("<TASK:QUESTION_OF_THE_DAY>", background_generator.calls[0][0])
        self.assertIn("Practical wisdom applied to action.", background_generator.calls[0][0])
        self.assertIn("reflection or judgment, not test recall", background_generator.calls[0][0])

    def test_daily_question_drops_an_unrecognized_insight_citation(self) -> None:
        generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "question": "What distinction would clarify the next step?",
                        "reason_for_asking": "A distinction seemed likely to open the inquiry.",
                        "cited_insight_title": "Invented Insight",
                    }
                )
            ]
        )
        service = AquinasGenerationService(generator, background_generator=generator)

        result = service.generate_daily_question(
            "Inquiry",
            [ConversationMessage(role="user", text="What comes next?")],
            [
                DailyQuestionInsight(
                    title="Real Insight",
                    definition="A grounded definition.",
                )
            ],
        )

        self.assertIsNone(result.cited_insight_title)

    def test_daily_question_uses_direct_json_background_generator(self) -> None:
        deep_background = RecordingGenerator([])
        fast_background = RecordingGenerator(
            [
                json.dumps(
                    {
                        "question": "Which unresolved distinction matters most?",
                        "reason_for_asking": "The conversation left one distinction open.",
                        "cited_insight_title": None,
                    }
                )
            ]
        )
        service = AquinasGenerationService(
            RecordingGenerator([]),
            background_generator=deep_background,
            background_fast_generator=fast_background,
        )

        result = service.generate_daily_question(
            "Inquiry",
            [ConversationMessage(role="user", text="What remains unresolved?")],
        )

        self.assertTrue(result.question.endswith("?"))
        self.assertEqual(deep_background.calls, [])
        self.assertEqual(len(fast_background.calls), 1)

    def test_daily_question_repairs_invalid_output_once(self) -> None:
        generator = RecordingGenerator(
            [
                "not json",
                json.dumps(
                    {
                        "question": "What assumption should you examine next?",
                        "reason_for_asking": "One assumption remains unresolved.",
                        "cited_insight_title": None,
                    }
                ),
            ]
        )
        service = AquinasGenerationService(generator, background_generator=generator)

        result = service.generate_daily_question(
            "Inquiry",
            [ConversationMessage(role="user", text="What comes next?")],
        )

        self.assertTrue(result.question.endswith("?"))
        self.assertEqual(len(generator.calls), 2)
        self.assertIn("<TASK:REPAIR_QUESTION_OF_THE_DAY>", generator.calls[1][0])

    def test_automatic_generation_mode_defaults_to_fast(self) -> None:
        mode = resolve_conversation_generation_mode(
            [ConversationMessage(role="user", text="What is prudence?")],
            ConversationGenerationMode.AUTOMATIC,
        )

        self.assertEqual(mode, ConversationGenerationMode.FAST)

    def test_automatic_generation_mode_routes_complex_questions_to_deep(self) -> None:
        explicit = resolve_conversation_generation_mode(
            [
                ConversationMessage(
                    role="user",
                    text="Compare and contrast prudence and wisdom.",
                )
            ],
            ConversationGenerationMode.AUTOMATIC,
        )
        multipart = resolve_conversation_generation_mode(
            [
                ConversationMessage(
                    role="user",
                    text="What is prudence? How does it guide action?",
                )
            ],
            ConversationGenerationMode.AUTOMATIC,
        )
        relational = resolve_conversation_generation_mode(
            [
                ConversationMessage(
                    role="user",
                    text="How do act, potency, form, and privation work together?",
                )
            ],
            ConversationGenerationMode.AUTOMATIC,
        )
        defeated_premise = resolve_conversation_generation_mode(
            [
                ConversationMessage(
                    role="user",
                    text="Doesn't that counterexample contradict your premise?",
                )
            ],
            ConversationGenerationMode.AUTOMATIC,
        )

        self.assertEqual(explicit, ConversationGenerationMode.DEEP)
        self.assertEqual(multipart, ConversationGenerationMode.DEEP)
        self.assertEqual(relational, ConversationGenerationMode.DEEP)
        self.assertEqual(defeated_premise, ConversationGenerationMode.DEEP)

    def test_successful_habit_counterexample_requires_explicit_revision(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [
                ConversationMessage(
                    role="user",
                    text="Every voluntary action must be explicitly chosen when it occurs.",
                ),
                ConversationMessage(
                    role="assistant",
                    text="Yes, every voluntary act requires a present explicit choice.",
                ),
                ConversationMessage(
                    role="user",
                    text=(
                        "A deliberately acquired habit can produce an act voluntary in "
                        "its cause without a fresh explicitly chosen act. Doesn't that "
                        "contradict the premise?"
                    ),
                ),
            ],
            personality=ConversationPersonality.SCHOLARLY,
        )

        self.assertIn(
            '"Yes. That contradicts my earlier universal claim."',
            prompt,
        )
        self.assertIn("some acts are voluntary through a present choice", prompt)
        self.assertIn("no more than three compact", prompt)
        self.assertIn("Do not call the contradiction apparent", prompt)
        self.assertIn("act of will as a final cause", prompt)

    def test_habit_revision_fails_closed_on_causal_category_error(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "response": "Yes. That contradicts my earlier universal claim. Some acts are voluntary through a present choice, while others can be voluntary in their cause through a relevant prior voluntary act even without a fresh explicit choice at execution.\\n\\nThe act of will is a final cause.",
                  "thinking_summary": [
                    "Distinguish present choice from voluntariness in cause.",
                    "Treat the act of will as a final cause."
                  ],
                  "key_terms": [{
                    "display_text": "final cause",
                    "canonical_term": "Final Cause",
                    "context_excerpt": "The act of will is a final cause."
                  }],
                  "insight": null
                }
                """
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.respond(
            [
                ConversationMessage(
                    role="user",
                    text="Every voluntary act must be explicitly chosen when it occurs.",
                ),
                ConversationMessage(
                    role="assistant",
                    text="Yes, every voluntary act requires a present explicit choice.",
                ),
                ConversationMessage(
                    role="user",
                    text=(
                        "An act from a deliberately acquired habit can be voluntary "
                        "in its cause without a fresh explicitly chosen act. Doesn't "
                        "that contradict the premise?"
                    ),
                ),
            ],
            thinking_enabled=True,
        )

        self.assertTrue(
            result.response.startswith(
                "Yes. That contradicts my earlier universal claim."
            )
        )
        self.assertNotIn("final cause", result.response.casefold())
        self.assertEqual(result.key_terms, ())
        self.assertEqual(
            result.thinking_summary,
            ("Distinguish present choice from voluntariness in cause.",),
        )

    def test_fast_conversation_mode_uses_fast_generator_and_summary_first_shape(self) -> None:
        deep_generator = RecordingGenerator([])
        fast_generator = RecordingGenerator(
            [
                '{"thinking_summary":["Named the central distinction.",'
                '"Applied it briefly."],"response":"Answer.","key_terms":[]}'
            ]
        )
        service = AquinasGenerationService(
            deep_generator,
            fast_generator=fast_generator,
        )

        result = service.respond(
            [ConversationMessage(role="user", text="Tell me briefly about prudence.")],
            thinking_enabled=True,
            generation_mode=ConversationGenerationMode.AUTOMATIC,
        )

        self.assertEqual(result.response, "Answer.")
        self.assertEqual(deep_generator.calls, [])
        prompt = fast_generator.calls[0][0]
        self.assertLess(
            prompt.index('"thinking_summary"'),
            prompt.index('"response"'),
        )
        self.assertIn("1-3 short, user-facing approach notes", prompt)
        self.assertIn("only as many as the inquiry", prompt)
        self.assertIn("specific to this inquiry", prompt)
        self.assertIn("concise active language", prompt)
        self.assertIn("Do not use generic", prompt)
        self.assertIn("reveal its conclusion prematurely", prompt)
        self.assertIn("tools that were not actually used", prompt)
        self.assertIn("neutral and independent", prompt)
        self.assertIn(
            "Let the question's scope and the user's requested format determine",
            prompt,
        )
        self.assertNotIn("250 words", prompt)
        self.assertIn("warm and seasoned guide", prompt)
        self.assertIn("lean into intentional dialogue", prompt)

    def test_tree_analysis_uses_background_generator(self) -> None:
        foreground_generator = RecordingGenerator([])
        background_generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "subject": {
                            "label": "Prudence",
                            "summary": "Practical reason.",
                        },
                        "insight_candidate": None,
                    }
                )
            ]
        )
        service = AquinasGenerationService(
            foreground_generator,
            background_generator=background_generator,
        )

        service.analyze_tree_update(
            "What is prudence?",
            "Prudence guides action according to right reason.",
        )

        self.assertEqual(foreground_generator.calls, [])
        self.assertEqual(len(background_generator.calls), 1)

    def test_extracts_only_the_central_grounded_tree_insight(self) -> None:
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
                  "insight_candidate": {
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
        self.assertEqual(
            result.insight_candidate.label,
            "Participation in Eternal Law",
        )
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
                        "insight_candidate": {
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

        self.assertIsNotNone(result.insight_candidate)
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
                    }
                )
            ]
        )

        subject = AquinasGenerationService(generator).label_tree_subject(
            ["Divine Will: God's willing is identical with the divine essence."]
        )

        self.assertEqual(subject.label, "Divine Providence")
        self.assertIn("<TASK:INSIGHT_TREE_NODE_SUBJECT>", generator.calls[0][0])
        self.assertIn("prefer one word", generator.calls[0][0])
        self.assertNotIn('"summary"', generator.calls[0][0])

    def test_accepts_single_word_node_subject(self) -> None:
        generator = RecordingGenerator([json.dumps({"label": "Virtue"})])

        subject = AquinasGenerationService(generator).label_tree_subject(
            ["Prudence: practical wisdom directing action."]
        )

        self.assertEqual(subject.label, "Virtue")

    def test_repairs_node_subject_that_repeats_insight_title(self) -> None:
        generator = RecordingGenerator(
            [
                json.dumps({"label": "Predestination"}),
                json.dumps({"label": "Divine Providence"}),
            ]
        )

        subject = AquinasGenerationService(generator).label_tree_subject(
            ["Predestination: God's ordering of creatures toward their final end."]
        )

        self.assertEqual(subject.label, "Divine Providence")
        self.assertEqual(len(generator.calls), 2)
        self.assertIn("must not repeat any Insight title", generator.calls[1][0])

    def test_repairs_node_subject_longer_than_five_words(self) -> None:
        generator = RecordingGenerator(
            [
                json.dumps({"label": "The General Foundations of Practical Moral Reasoning"}),
                json.dumps({"label": "Practical Reason"}),
            ]
        )

        subject = AquinasGenerationService(generator).label_tree_subject(
            ["Prudence: practical wisdom directing action."]
        )

        self.assertEqual(subject.label, "Practical Reason")
        self.assertIn(
            "<TASK:REPAIR_INSIGHT_TREE_NODE_SUBJECT>",
            generator.calls[1][0],
        )

    def test_repairs_punctuation_only_node_subject(self) -> None:
        generator = RecordingGenerator(
            [
                '{"label":"..."}',
                '{"label":"Practical Reason"}',
            ]
        )

        subject = AquinasGenerationService(generator).label_tree_subject(
            [
                "Synderesis. First practical principles.",
                "Conscience. Particular practical judgment.",
                "Prudence. Right reason about action.",
            ]
        )

        self.assertEqual(subject.label, "Practical Reason")
        self.assertEqual(len(generator.calls), 2)

    def test_generates_weighted_midpoint_concept(self) -> None:
        candidate_payloads = [
            {
                "title": "Prudential Natural Law",
                "part_of_speech": "",
                "pronunciation": "",
                "definition": (
                    "Prudence applies natural-law principles to the "
                    "particular demands of action."
                ),
                "example": "A judge applies a general norm equitably.",
            }
        ] + [
            {
                "title": f"Distinct Synthesis {index}",
                "part_of_speech": "",
                "pronunciation": "",
                "definition": (
                    "Prudence applies Natural Law in a distinct synthesis "
                    f"{index}."
                ),
                "example": "",
            }
            for index in range(2, 6)
        ]
        generator = RecordingGenerator(
            [json.dumps({"candidates": candidate_payloads})]
        )
        concepts = [
            ContextualDefinition(
                title="Prudence",
                part_of_speech="",
                pronunciation="",
                definition="Practical reason directing action.",
                example="",
            ),
            ContextualDefinition(
                title="Natural Law",
                part_of_speech="",
                pronunciation="",
                definition="Rational participation in eternal law.",
                example="",
            ),
        ]

        candidates = AquinasGenerationService(generator).blend_concept_candidates(
            concepts,
            [3.0, 1.0],
        )

        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0].title, "Prudential Natural Law")
        prompt, limit = generator.calls[0]
        self.assertIn("<TASK:MIDPOINT_CONCEPT_BLEND>", prompt)
        self.assertIn('"weight_percent": 75.0', prompt)
        self.assertIn('"weight_percent": 25.0', prompt)
        self.assertIn("weighted embedding centroid", prompt)
        self.assertEqual(limit, MIDPOINT_GENERATION_MAX_TOKENS)

    def test_midpoint_rejects_more_than_eight_concepts(self) -> None:
        concepts = [
            ContextualDefinition(
                title=f"Concept {index}",
                part_of_speech="",
                pronunciation="",
                definition=f"Definition {index}.",
                example="",
            )
            for index in range(9)
        ]

        with self.assertRaisesRegex(
            StructuredGenerationError,
            "at most eight",
        ):
            AquinasGenerationService(RecordingGenerator([])).blend_concept_candidates(
                concepts,
                [1.0] * 9,
            )

    def test_high_source_midpoint_builds_integrated_center_brief(self) -> None:
        sources = tuple(
            (
                ContextualDefinition(
                    title=title,
                    part_of_speech="",
                    pronunciation="",
                    definition=f"{title} definition.",
                    example="",
                ),
                weight,
            )
            for title, weight in (
                ("Prudence", 0.4),
                ("Justice", 0.2),
                ("Temperance", 0.15),
                ("Fortitude", 0.15),
                ("Law", 0.1),
            )
        )

        prompt = AquinasGenerationService._midpoint_center_prompt(sources)

        self.assertIn("grammatical and conceptual core", prompt)
        self.assertIn("one subordinate functional role", prompt)
        self.assertIn("not a list, checklist", prompt)
        self.assertIn('"weight_percent": 40.0', prompt)
        self.assertEqual(
            AquinasGenerationService._parse_midpoint_center(
                '{"center":"Prudence and justice organize action."}'
            ),
            "Prudence and justice organize action.",
        )

    def test_high_source_midpoint_requires_every_source_in_each_candidate(self) -> None:
        sources = [
            ContextualDefinition(
                title=title,
                part_of_speech="",
                pronunciation="",
                definition=f"{title} definition.",
                example="",
            )
            for title in ("Prudence", "Justice", "Temperance", "Fortitude", "Law")
        ]
        candidates = [
            ContextualDefinition(
                title=f"Candidate {index}",
                part_of_speech="",
                pronunciation="",
                definition="Prudence, justice, temperance, and fortitude cooperate.",
                example="",
            )
            for index in range(5)
        ]

        with self.assertRaisesRegex(
            StructuredGenerationError,
            "missing: Law",
        ):
            AquinasGenerationService._validate_midpoint_source_coverage(
                candidates,
                tuple((source, 1.0) for source in sources),
            )

    def test_two_source_midpoint_requires_both_sources_in_every_candidate(self) -> None:
        justice = ContextualDefinition(
            title="Justice",
            part_of_speech="",
            pronunciation="",
            definition="Rendering to each what is due.",
            example="",
        )
        mercy = ContextualDefinition(
            title="Mercy",
            part_of_speech="",
            pronunciation="",
            definition="Compassion expressed through fitting aid.",
            example="",
        )
        candidates = [
            ContextualDefinition(
                title=f"Candidate {index}",
                part_of_speech="",
                pronunciation="",
                definition=(
                    "Justice renders what is due while mercy offers fitting aid."
                    if index
                    else "Justice renders what is due."
                ),
                example="",
            )
            for index in range(5)
        ]

        with self.assertRaisesRegex(
            StructuredGenerationError,
            "missing: Mercy",
        ):
            AquinasGenerationService._validate_midpoint_source_coverage(
                candidates,
                ((justice, 0.5), (mercy, 0.5)),
            )

    def test_repairs_invalid_midpoint_json(self) -> None:
        valid_candidates = [
            {
                "title": "Practical Moral Judgment",
                "part_of_speech": "",
                "pronunciation": "",
                "definition": (
                    "Natural Law's universal moral truth is applied through "
                    "prudence."
                ),
                "example": "",
            }
        ] + [
            {
                "title": f"Repaired Candidate {index}",
                "part_of_speech": "",
                "pronunciation": "",
                "definition": (
                    "Prudence applies Natural Law principles in a repaired "
                    f"synthesis {index}."
                ),
                "example": "",
            }
            for index in range(2, 6)
        ]
        generator = RecordingGenerator(
            [
                "Prudence and natural law guide concrete action.",
                json.dumps({"candidates": valid_candidates}),
            ]
        )
        concepts = [
            ContextualDefinition("Prudence", "", "", "Practical reason.", ""),
            ContextualDefinition("Natural Law", "", "", "Moral first principles.", ""),
        ]

        candidates = AquinasGenerationService(generator).blend_concept_candidates(
            concepts,
            [0.5, 0.5],
        )

        self.assertEqual(candidates[0].title, "Practical Moral Judgment")
        self.assertIn(
            "<TASK:REPAIR_MIDPOINT_CONCEPT_BLEND>",
            generator.calls[1][0],
        )
        self.assertIn(
            "Source integration overrides preservation",
            generator.calls[1][0],
        )

    def test_generates_exactly_three_make_node_children(self) -> None:
        generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "children": [
                            {
                                "title": "Habit of Right Reason",
                                "definition": "Prudence is strengthened through repeated right judgment.",
                                "example": "A physician learns to judge cases well through practice.",
                            },
                            {
                                "title": "Prudence and Moral Virtue",
                                "definition": "Prudence directs virtues toward fitting action.",
                                "example": "Courage follows prudent judgment rather than impulse.",
                            },
                            {
                                "title": "Judgment in Particulars",
                                "definition": "Prudence applies universal goods to concrete circumstances.",
                                "example": "A general duty is adapted to an urgent situation.",
                            },
                        ]
                    }
                )
            ]
        )
        concept = ContextualDefinition(
            title="Prudence",
            part_of_speech="",
            pronunciation="",
            definition="Practical reason directing action.",
            example="",
        )

        children = AquinasGenerationService(generator).generate_concept_children(
            concept
        )

        self.assertEqual(len(children), 3)
        self.assertEqual(children[0].title, "Habit of Right Reason")
        prompt, limit = generator.calls[0]
        self.assertIn("<TASK:MAKE_NODE_CHILDREN>", prompt)
        self.assertIn('"title": "Prudence"', prompt)
        self.assertIn("Adapt the expansion to the concept", prompt)
        self.assertIn("three most illuminating dimensions", prompt)
        self.assertIn("non-overlapping", prompt)
        self.assertIn("Do not add filler", prompt)
        self.assertIn("new claim, question, distinction", prompt)
        self.assertIn("Definition, Conditions, Assessment", prompt)
        self.assertEqual(limit, MAKE_NODE_GENERATION_MAX_TOKENS)

    def test_repairs_invalid_make_node_children(self) -> None:
        valid_children = {
            "children": [
                {"title": "First Child", "definition": "First definition.", "example": ""},
                {"title": "Second Child", "definition": "Second definition.", "example": ""},
                {"title": "Third Child", "definition": "Third definition.", "example": ""},
            ]
        }
        generator = RecordingGenerator(
            [
                '{"children":[{"title":"Only One","definition":"Incomplete."}]}',
                json.dumps(valid_children),
            ]
        )
        concept = ContextualDefinition(
            title="Prudence",
            part_of_speech="",
            pronunciation="",
            definition="Practical reason directing action.",
            example="",
        )

        children = AquinasGenerationService(generator).generate_concept_children(
            concept
        )

        self.assertEqual(len(children), 3)
        self.assertIn(
            "<TASK:REPAIR_MAKE_NODE_CHILDREN>",
            generator.calls[1][0],
        )
        repair_prompt = generator.calls[1][0]
        self.assertIn("Do not reuse the parent title", repair_prompt)
        self.assertIn("Moral Object", repair_prompt)
        self.assertIn("Means-End Order", repair_prompt)
        self.assertIn("Proportionate Reason", repair_prompt)

    def test_discards_ungrounded_tree_insight_candidate(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "subject": {"label": "Grace", "summary": "A summary."},
                  "insight_candidate": {
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

        self.assertIsNone(result.insight_candidate)

    def test_discards_acknowledgement_and_procedural_tree_candidates(self) -> None:
        outputs = iter(
            [
                json.dumps(
                    {
                        "subject": {"label": "Courtesy", "summary": "A reply."},
                        "insight_candidate": {
                            "label": "Polite Acknowledgement",
                            "summary": "The assistant acknowledges gratitude.",
                            "evidence_excerpt": "You're welcome!",
                        },
                    }
                ),
                json.dumps(
                    {
                        "subject": {"label": "Interface Steps", "summary": "A step."},
                        "insight_candidate": {
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

        self.assertIsNone(acknowledgement.insight_candidate)
        self.assertIsNone(procedural.insight_candidate)
        self.assertEqual(acknowledgement.subject_label, "")
        self.assertEqual(acknowledgement.subject_summary, "")
        self.assertEqual(procedural.subject_label, "")
        self.assertEqual(procedural.subject_summary, "")

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
        self.assertIn("Preserve attribution and epistemic status", generator.calls[0][0])
        self.assertIn("what the assistant proposed", generator.calls[0][0])
        self.assertIn("what remains tentative, disputed, uncertain", generator.calls[0][0])
        self.assertIn("into a settled fact", generator.calls[0][0])
        self.assertIn("the earlier claim, the revised position", generator.calls[0][0])
        self.assertIn("decisive reason", generator.calls[0][0])
        self.assertIn("revised position as current", generator.calls[0][0])
        self.assertEqual(
            generator.calls[0][1],
            COMPACTION_GENERATION_MAX_TOKENS,
        )

    def test_compaction_repairs_invalid_output_once(self) -> None:
        generator = RecordingGenerator(
            [
                "not json",
                '{"summary":"The user revised the earlier claim after identifying a contradiction."}',
            ]
        )
        service = AquinasGenerationService(generator)

        summary = service.compact_context(
            [ConversationMessage(role="user", text="I changed my mind.")],
            compacted_context="The earlier claim remained tentative.",
        )

        self.assertIn("revised", summary)
        self.assertEqual(len(generator.calls), 2)
        self.assertIn(
            "<TASK:REPAIR_COMPACT_CONVERSATION_CONTEXT>",
            generator.calls[1][0],
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

    def test_conversation_prompt_describes_supported_block_markdown(self) -> None:
        service = AquinasGenerationService(
            RecordingGenerator(
                ['{"response":"Answer.","thinking_summary":[],"key_terms":[]}']
            )
        )

        prompt = service.conversation_prompt(
            [ConversationMessage(role="user", text="Explain the structure.")]
        )

        self.assertIn('H1 on its own line with "# "', prompt)
        self.assertIn('H2 on its own line with "## "', prompt)
        self.assertIn('unordered-list item on its own line with "- "', prompt)
        self.assertIn('ordered-list item on its own line with "1. "', prompt)
        self.assertIn("line break as \\n inside the JSON string", prompt)
        self.assertIn("set insight to a complete object", prompt)
        self.assertIn('"insight": null', prompt)
        self.assertIn("Intellectually foundational concepts or subjects", prompt)
        self.assertIn("prerequisite knowledge", prompt)
        self.assertIn("There is no highlighting quota", prompt)
        self.assertIn("never introduce jargon", prompt)
        self.assertIn("Discovery concepts", prompt)
        self.assertIn("intellectual", prompt)
        self.assertIn("rabbit hole", prompt)
        self.assertIn("Never invent a citation", prompt)
        self.assertIn("level of explanation the user actually requested", prompt)
        self.assertIn("Do not add divine, theological, or devotional framing", prompt)
        self.assertIn("Merely naming Aquinas", prompt)
        self.assertIn("does not license an opening reference to God", prompt)
        self.assertIn("Do not collapse", prompt)
        self.assertIn("closely related concepts into synonyms", prompt)
        self.assertIn("distinctive role", prompt)
        self.assertIn("change actualizes a prior potency", prompt)
        self.assertIn("subject capable of receiving it", prompt)
        self.assertIn("identify form with essence", prompt)
        self.assertIn("unshaped bronze", prompt)
        self.assertIn("Never use a mineral becoming a plant", prompt)
        self.assertIn("keep intellect and will distinct", prompt)
        self.assertIn("intellect apprehends, deliberates, and judges", prompt)
        self.assertIn("Choice is an act of will", prompt)
        self.assertIn("The intended end is a final cause", prompt)
        self.assertIn("Voluntariness in cause", prompt)
        self.assertIn("equally or automatically voluntary", prompt)
        self.assertIn("adjacent concept naturally contributes", prompt)
        self.assertIn("do not manufacture jargon", prompt)
        self.assertIn("Treat alternate wording and close synonyms as one concept", prompt)
        self.assertIn("both act and actuality", prompt)
        self.assertIn("broad topic word", prompt)

    def test_direct_definition_prompt_displays_required_insight_shape(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [ConversationMessage(role="user", text="What does prudence mean?")],
            thinking_enabled=True,
            generation_mode=ConversationGenerationMode.FAST,
        )

        self.assertIn('"title": "prudence"', prompt)
        self.assertIn('"definition": "A concise contextual definition."', prompt)
        self.assertNotIn('"insight": null', prompt)
        self.assertIn("not novelty alone", prompt)
        self.assertIn("immediate conversation or passage's specific sense", prompt)
        self.assertIn("Prefer one precise", prompt)
        self.assertIn("distinction over a list of loose synonyms", prompt)
        self.assertIn("ambiguity briefly in response", prompt)
        self.assertIn("independent of the", prompt)
        self.assertIn("selected conversational persona", prompt)
        self.assertIn("renders the complete in-text Insight card", prompt)
        self.assertIn("before response", prompt)
        self.assertIn("must still answer the user's inquiry after that card", prompt)
        self.assertIn("do not reduce it to", prompt)
        self.assertIn(
            '"response": "A concise explanation of how prudence functions '
            'in the requested context."',
            prompt,
        )

    def test_direct_definition_repair_preserves_contextual_precision(self) -> None:
        prompt = AquinasGenerationService._conversation_repair_prompt(
            '{"response":"I found it.","thinking_summary":[],"key_terms":[]}',
            thinking_enabled=True,
            required_insight_term="participation",
        )

        self.assertIn("immediate conversation or passage's specific sense", prompt)
        self.assertIn("Prefer one precise", prompt)
        self.assertIn("distinction over loose synonyms", prompt)
        self.assertIn("do not invent an author", prompt)

    def test_direct_definition_preserves_substantive_answer_after_insight(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "response": "Participation means receiving existence in a limited way.",
                  "thinking_summary": ["Identify the contextual contrast."],
                  "key_terms": [{
                    "display_text": "Participation",
                    "canonical_term": "Participation",
                    "context_excerpt": "Participation means receiving existence in a limited way."
                  }],
                  "insight": {
                    "title": "participation",
                    "definition": "Receiving existence in a limited rather than essential way."
                  }
                }
                """
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.respond(
            [ConversationMessage(role="user", text="What does participation mean here?")],
            thinking_enabled=True,
            personality=ConversationPersonality.FUN,
        )

        self.assertEqual(
            result.response,
            "Participation means receiving existence in a limited way.",
        )
        self.assertEqual(len(result.key_terms), 1)
        self.assertIsNotNone(result.insight)

    def test_direct_definition_preserves_context_ambiguity_warning(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        result = service.parse_conversation_output(
            """
            {
              "response": "The context is ambiguous, so this could mean either kind of participation.",
              "thinking_summary": [],
              "key_terms": [],
              "insight": {
                "title": "participation",
                "definition": "Taking part in or sharing in something."
              }
            }
            """,
            required_insight_term="participation",
        )

        self.assertIn("ambiguous", result.response)

    def test_scholarly_personality_is_thomistic_rigorous_and_warm(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [
                ConversationMessage(
                    role="user",
                    text="How should justice guide a difficult decision?",
                )
            ],
            personality=ConversationPersonality.SCHOLARLY,
        )

        self.assertIn("wise, learned, and well-spoken mentor", prompt)
        self.assertIn("Thomistic intellectual tradition", prompt)
        self.assertIn("scholarly rigor with humane warmth", prompt)
        self.assertIn("respected student and fellow inquirer", prompt)
        self.assertIn("nature, causes, purpose", prompt)
        self.assertIn("strongest reasonable form", prompt)
        self.assertIn("ease of a generous teacher", prompt)
        self.assertIn("measured gravity without stiffness", prompt)
        self.assertIn("do not force theological framing", prompt)
        self.assertIn("Avoid\narchaic imitation, coldness, condescension", prompt)

    def test_balanced_personality_is_warm_and_can_deepen_the_dialogue(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [ConversationMessage(role="user", text="Help me think this through.")],
            personality=ConversationPersonality.BALANCED,
        )

        self.assertIn("warm and seasoned guide", prompt)
        self.assertIn("hospitable", prompt)
        self.assertIn("Match the depth", prompt)
        self.assertIn("lean into intentional dialogue", prompt)
        self.assertIn("ask a focused question", prompt)
        self.assertIn("do not force a formal dialectic", prompt)

    def test_socratic_personality_guides_without_withholding_answers(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [
                ConversationMessage(
                    role="user",
                    text="How should justice guide a difficult decision?",
                )
            ],
            personality=ConversationPersonality.SOCRATIC,
        )

        self.assertIn("thoughtful Socratic guide", prompt)
        self.assertIn("one meaningful question at", prompt)
        self.assertIn("a time, chosen to move the inquiry forward", prompt)
        self.assertIn("Do not", prompt)
        self.assertIn("merely withhold answers", prompt)
        self.assertIn("simple factual, practical, or urgent requests", prompt)

    def test_fun_personality_is_personable_without_dismissing_serious_topics(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [ConversationMessage(role="user", text="Help me understand this.")],
            personality=ConversationPersonality.FUN,
        )

        self.assertIn("user's best friend", prompt)
        self.assertIn("light slang naturally", prompt)
        self.assertIn("eccentric", prompt)
        self.assertIn("Remain respectful and kind at all times", prompt)
        self.assertIn("familiarity must never", prompt)
        self.assertIn("become flippancy", prompt)
        self.assertIn("Never force slang", prompt)
        self.assertIn("Match the seriousness of the moment", prompt)
        self.assertIn("without abandoning the casual", prompt)
        self.assertIn("Let humor and eccentricity recede", prompt)
        self.assertIn("make the user feel dismissed", prompt)

    def test_connection_inquiry_uses_neutral_set_analysis(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))
        prompt = service.conversation_prompt(
            [
                ConversationMessage(
                    role="user",
                    text=(
                        "<inquire_connection>\n"
                        "- Prudence: Right reason applied to action.\n"
                        "- Natural Law: Rational participation in eternal law.\n"
                        "- Equity: Justice adapted to particular circumstances.\n"
                        "</inquire_connection>\n\n"
                        "How do these guide judgment?"
                    ),
                )
            ]
        )

        self.assertIn("neutral application Insight action", prompt)
        self.assertIn("independent of the", prompt)
        self.assertIn("selected conversational persona", prompt)
        self.assertIn("strongest meaningful relationship", prompt)
        self.assertIn("do not force a synthesis", prompt)
        self.assertIn("than two concepts", prompt)
        self.assertIn("organizing pattern across the set", prompt)
        self.assertIn("preserve the kind of thing each selection", prompt)
        self.assertIn("Treat the supplied descriptions as the source of truth", prompt)
        self.assertIn("type-bearing head nouns", prompt)
        self.assertIn("judgment rather", prompt)
        self.assertIn("than a faculty", prompt)
        self.assertIn("habit, do not reduce it to mere potential", prompt)
        self.assertIn("virtue, do not", prompt)
        self.assertIn("turn it into an executive power", prompt)
        self.assertIn("Distinguish cooperation, application", prompt)
        self.assertIn("hierarchy or linear pipeline only when", prompt)
        self.assertIn("complementary roles", prompt)
        self.assertIn('"foundation, application, and governance"', prompt)
        self.assertIn("synderesis is a habit of first practical principles", prompt)
        self.assertIn("conscience is the act or judgment", prompt)
        self.assertIn("prudence is the virtue perfecting practical", prompt)
        self.assertIn("one to three", prompt)
        self.assertIn("compact paragraphs", prompt)
        self.assertIn("first explain briefly", prompt)
        self.assertIn("selected from the Tree for relationship analysis", prompt)
        self.assertIn("then perform the analysis", prompt)
        self.assertIn("Do not stop after explaining the chip", prompt)
        self.assertEqual(
            resolve_conversation_generation_mode(
                [
                    ConversationMessage(
                        role="user",
                        text="<inquire_connection>...</inquire_connection>",
                    )
                ],
                ConversationGenerationMode.AUTOMATIC,
            ),
            ConversationGenerationMode.DEEP,
        )

    def test_connection_inquiry_marker_is_case_insensitive(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))

        prompt = service.conversation_prompt(
            [
                ConversationMessage(
                    role="user",
                    text="<INQUIRE_CONNECTION>Prudence and conscience",
                )
            ],
            personality=ConversationPersonality.FUN,
        )

        self.assertIn("neutral application Insight action", prompt)
        self.assertNotIn("user's best friend", prompt)

    def test_conversation_prompt_defines_tree_controls_authoritatively(self) -> None:
        service = AquinasGenerationService(RecordingGenerator([]))
        prompt = service.conversation_prompt(
            [
                ConversationMessage(
                    role="user",
                    text="What do the Midpoint and Make Node buttons do?",
                )
            ]
        )

        self.assertIn("Do not mention these controls unless the user asks", prompt)
        self.assertIn("2-8 selected Insights or Node Concepts", prompt)
        self.assertIn("weighted centroid from their embedding vectors", prompt)
        self.assertIn("nearest the centroid", prompt)
        self.assertIn("connected to its sources", prompt)
        self.assertIn("not merely", prompt)
        self.assertIn("verbal 50/50 mixture", prompt)
        self.assertIn("promotes the selected Insight into a Node Concept", prompt)
        self.assertIn("exactly three", prompt)
        self.assertIn("distinct, non-overlapping child Insights", prompt)
        self.assertIn("maximized angular separation", prompt)
        self.assertIn("existing\n  connections", prompt)
        self.assertIn("brief plain-language explanation", prompt)

    def test_parses_requested_inline_insight(self) -> None:
        service = AquinasGenerationService(
            RecordingGenerator(
                [
                    """
                    {
                      "response": "Here is an Insight for prudence.",
                      "thinking_summary": [],
                      "key_terms": [],
                      "insight": {
                        "title": "Prudence",
                        "definition": "Right reason applied to action."
                      }
                    }
                    """
                ]
            )
        )

        result = service.respond(
            [
                ConversationMessage(
                    role="user",
                    text="Make me an Insight for prudence.",
                )
            ]
        )

        self.assertIsNotNone(result.insight)
        assert result.insight is not None
        self.assertEqual(result.insight.title, "Prudence")
        self.assertEqual(
            result.insight.definition,
            "Right reason applied to action.",
        )
        self.assertEqual(result.insight.part_of_speech, "")
        self.assertEqual(result.insight.pronunciation, "")
        self.assertEqual(result.insight.example, "")

    def test_direct_definition_request_requires_inline_insight(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "response": "Prudence directs action according to right reason.",
                  "thinking_summary": [],
                  "key_terms": [],
                  "insight": null
                }
                """,
                """
                {
                  "response": "Here is a definition of prudence.",
                  "thinking_summary": [],
                  "key_terms": [],
                  "insight": {
                    "title": "Prudence",
                    "definition": "Right reason applied to action."
                  }
                }
                """,
            ]
        )
        service = AquinasGenerationService(generator)

        result = service.respond(
            [ConversationMessage(role="user", text="What is prudence?")]
        )

        self.assertEqual(result.insight.title, "Prudence")
        self.assertEqual(len(generator.calls), 2)
        self.assertIn(
            'directly asks for a definition of "prudence"',
            generator.calls[0][0],
        )
        self.assertIn(
            "insight must",
            generator.calls[1][0],
        )
        self.assertIn(
            "must not be null",
            generator.calls[1][0],
        )

    def test_extracts_terms_only_from_direct_definition_requests(self) -> None:
        cases = {
            "Define analogical predication.": "analogical predication",
            "What does hypostasis mean?": "hypostasis",
            "Give me a definition of natural law.": "natural law",
            "What are universals?": "universals",
            "What is the meaning of subsidiarity?": "subsidiarity",
            "What is meant by act and potency?": "act and potency",
            "Could you define hylomorphism?": "hylomorphism",
            "Can you tell me what teleology means?": "teleology",
            "Please explain the concept common good.": "common good",
            "What does participation mean here?": "participation",
            "What does participation mean in this passage?": "participation",
            "Could you define participation in this context?": "participation",
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(
                    requested_definition_term(
                        [ConversationMessage(role="user", text=question)]
                    ),
                    expected,
                )

        self.assertIsNone(
            requested_definition_term(
                [
                    ConversationMessage(
                        role="user",
                        text="How does grace relate to nature?",
                    )
                ]
            )
        )
        self.assertIsNone(
            requested_definition_term(
                [
                    ConversationMessage(
                        role="user",
                        text="Explain how grace relates to nature.",
                    )
                ]
            )
        )
        self.assertIsNone(
            requested_definition_term(
                [
                    ConversationMessage(
                        role="user",
                        text="What is the difference between essence and existence?",
                    )
                ]
            )
        )
        for question in (
            "What is the Midpoint button?",
            "What does this chip mean?",
            "Define the Make Node control.",
            "What does Make Node mean?",
            "What is Inquire Connection?",
        ):
            with self.subTest(interface_question=question):
                self.assertIsNone(
                    requested_definition_term(
                        [ConversationMessage(role="user", text=question)]
                    )
                )

    def test_conversation_images_are_labeled_and_forwarded_to_generation(self) -> None:
        generator = RecordingGenerator(
            [
                json.dumps(
                    {
                        "response": "The image shows a handwritten syllogism.",
                        "thinking_summary": [],
                        "key_terms": [],
                        "insight": None,
                    }
                )
            ]
        )
        service = AquinasGenerationService(generator)
        image_data = b"test-image-bytes"

        result = service.respond(
            [
                ConversationMessage(
                    role="user",
                    text="What is this?",
                    images=(
                        ConversationImage(
                            name="Notes.jpg",
                            media_type="image/jpeg",
                            data=image_data,
                        ),
                    ),
                )
            ]
        )

        self.assertEqual(result.response, "The image shows a handwritten syllogism.")
        self.assertEqual(generator.image_calls, [(image_data,)])
        self.assertIn(
            "[Attached image 1: Notes.jpg (image/jpeg)]",
            generator.calls[0][0],
        )
        self.assertIn(
            "Inspect them directly",
            generator.calls[0][0],
        )
        self.assertIsNone(
            requested_definition_term(
                [
                    ConversationMessage(
                        role="user",
                        text="What is this?",
                        images=(
                            ConversationImage(
                                name="Notes.jpg",
                                media_type="image/jpeg",
                                data=image_data,
                            ),
                        ),
                    )
                ]
            )
        )

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

    def test_final_response_preserves_heading_line_boundary(self) -> None:
        generator = RecordingGenerator(
            [
                '{"response":"# The Headline\\nBody text remains a paragraph.",'
                '"thinking_summary":[],"key_terms":[]}'
            ]
        )
        service = AquinasGenerationService(generator)

        response = service.respond(
            [ConversationMessage(role="user", text="Explain this.")]
        )

        self.assertEqual(
            response.response,
            "# The Headline\nBody text remains a paragraph.",
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
            [
                ConversationMessage(
                    role="user",
                    text="How does natural law participate in eternal law?",
                )
            ],
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

    def test_limits_thinking_summary_to_three_notes(self) -> None:
        service = AquinasGenerationService(
            RecordingGenerator(
                [
                    """
                    {
                      "response": "Answer.",
                      "thinking_summary": ["One.", "Two.", "Three.", "Four."],
                      "key_terms": []
                    }
                    """
                ]
            )
        )

        result = service.respond(
            [ConversationMessage(role="user", text="Compare these accounts.")],
            thinking_enabled=True,
        )

        self.assertEqual(result.thinking_summary, ("One.", "Two.", "Three."))

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
            "user-facing approach summary is disabled",
            generator.calls[0][0],
        )
        self.assertIn(
            "Return an empty thinking_summary",
            generator.calls[0][0],
        )

    def test_parses_contextual_definition_json(self) -> None:
        generator = RecordingGenerator(
            [
                """
                {
                  "title": "Natural Law",
                  "context": "Moral knowledge",
                  "definition": "Natural law is reason's participation in eternal law."
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
        self.assertEqual(result.context, "Moral knowledge")
        self.assertEqual(result.part_of_speech, "")
        self.assertEqual(result.pronunciation, "")
        self.assertEqual(result.example, "")
        self.assertIn("eternal law", result.definition)
        self.assertEqual(len(generator.calls), 1)
        self.assertIn("<TASK:CONTEXTUAL_DEFINITION>", generator.calls[0][0])
        self.assertNotIn('"part_of_speech"', generator.calls[0][0])
        self.assertNotIn('"pronunciation"', generator.calls[0][0])
        self.assertNotIn('"example"', generator.calls[0][0])
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
                  "context": "Metaphysical composition",
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
                  "context": "Metaphysical composition",
                  "definition": "Essence describes what a thing is."
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
