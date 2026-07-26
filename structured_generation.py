"""Validated structured tasks performed by the Aquinas language model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence


Generator = Callable[[str, int], str]
STRUCTURED_GENERATION_MAX_TOKENS = 1_000
CONVERSATION_GENERATION_MAX_TOKENS = 1_800
COMPACTION_GENERATION_MAX_TOKENS = 1_200
TREE_ANALYSIS_MAX_TOKENS = 1_000


class StructuredGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    text: str


@dataclass(frozen=True)
class ContextualDefinition:
    title: str
    part_of_speech: str
    pronunciation: str
    definition: str
    example: str


@dataclass(frozen=True)
class GeneratedKeyTerm:
    display_text: str
    canonical_term: str
    context_excerpt: str


@dataclass(frozen=True)
class StructuredConversationResponse:
    response: str
    thinking_summary: tuple[str, ...]
    key_terms: tuple[GeneratedKeyTerm, ...]


@dataclass(frozen=True)
class GeneratedTreeNodeSeed:
    label: str
    summary: str
    evidence_excerpt: str


@dataclass(frozen=True)
class GeneratedTreeUpdate:
    subject_label: str
    subject_summary: str
    node_seed: GeneratedTreeNodeSeed | None


@dataclass(frozen=True)
class GeneratedNodeSubject:
    label: str
    summary: str


@dataclass(frozen=True)
class ConversationStreamUpdate:
    kind: str
    value: str | tuple[str, ...]


class ConversationStreamParser:
    """Incrementally exposes approved fields from structured model JSON.

    The parser intentionally ignores every other part of the raw generation,
    including any checkpoint-specific hidden thought channel.
    """

    def __init__(self) -> None:
        self._output = ""
        self._summary: tuple[str, ...] = ()
        self._response = ""

    def feed(self, fragment: str) -> tuple[ConversationStreamUpdate, ...]:
        self._output += fragment
        updates: list[ConversationStreamUpdate] = []

        summary = self._completed_string_array("thinking_summary")[:4]
        if summary != self._summary:
            self._summary = summary
            updates.append(
                ConversationStreamUpdate("thinking_summary", summary)
            )

        response = self._partial_string_field("response")
        if response.startswith(self._response) and len(response) > len(self._response):
            updates.append(
                ConversationStreamUpdate(
                    "response_delta",
                    response[len(self._response):],
                )
            )
            self._response = response

        return tuple(updates)

    def _completed_string_array(self, key: str) -> tuple[str, ...]:
        match = self._field_match(key, "[")
        if match is None:
            return ()

        values: list[str] = []
        index = match
        while index < len(self._output):
            character = self._output[index]
            if character == "]":
                break
            if character != '"':
                index += 1
                continue

            end = self._string_end(index)
            if end is None:
                break
            try:
                value = json.loads(self._output[index:end + 1])
            except json.JSONDecodeError:
                break
            if isinstance(value, str):
                cleaned = " ".join(value.split())
                if cleaned:
                    values.append(cleaned)
            index = end + 1

        return tuple(values)

    def _partial_string_field(self, key: str) -> str:
        start = self._field_match(key, '"')
        if start is None:
            return ""

        end = self._string_end(start - 1)
        raw = self._output[start:end] if end is not None else self._output[start:]
        for trim_count in range(0, min(6, len(raw)) + 1):
            candidate = raw if trim_count == 0 else raw[:-trim_count]
            try:
                value = json.loads(f'"{candidate}"')
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            return value if isinstance(value, str) else ""
        return ""

    def _field_match(self, key: str, opening_character: str) -> int | None:
        marker = f'"{key}"'
        marker_index = self._output.find(marker)
        if marker_index < 0:
            return None
        colon_index = self._output.find(":", marker_index + len(marker))
        if colon_index < 0:
            return None
        opening_index = colon_index + 1
        while (
            opening_index < len(self._output)
            and self._output[opening_index].isspace()
        ):
            opening_index += 1
        if (
            opening_index >= len(self._output)
            or self._output[opening_index] != opening_character
        ):
            return None
        return opening_index + 1

    def _string_end(self, opening_quote_index: int) -> int | None:
        escaped = False
        for index in range(opening_quote_index + 1, len(self._output)):
            character = self._output[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                return index
        return None


class AquinasGenerationService:
    def __init__(self, generator: Generator) -> None:
        self.generator = generator

    def define_term(
        self,
        term: str,
        source_excerpt: str = "",
        recent_messages: Sequence[ConversationMessage] = (),
    ) -> ContextualDefinition:
        prompt = self._definition_prompt(
            term=term,
            source_excerpt=source_excerpt,
            recent_messages=recent_messages,
        )
        first_output = self.generator(prompt, STRUCTURED_GENERATION_MAX_TOKENS)

        try:
            return self._parse_definition(first_output, fallback_title=term)
        except StructuredGenerationError:
            repair_prompt = self._repair_prompt(
                term=term,
                invalid_output=first_output,
            )
            repaired_output = self.generator(
                repair_prompt,
                STRUCTURED_GENERATION_MAX_TOKENS,
            )
            return self._parse_definition(repaired_output, fallback_title=term)

    def respond(
        self,
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
        thinking_enabled: bool = False,
    ) -> StructuredConversationResponse:
        if not recent_messages:
            raise StructuredGenerationError(
                "At least one conversation message is required."
            )

        prompt = self.conversation_prompt(
            recent_messages,
            compacted_context=compacted_context,
            thinking_enabled=thinking_enabled,
        )
        first_output = self.generator(
            prompt,
            CONVERSATION_GENERATION_MAX_TOKENS,
        )
        try:
            result = self._parse_conversation_response(first_output)
        except StructuredGenerationError:
            repaired_output = self.generator(
                self._conversation_repair_prompt(
                    first_output,
                    thinking_enabled=thinking_enabled,
                ),
                CONVERSATION_GENERATION_MAX_TOKENS,
            )
            result = self._parse_conversation_response(repaired_output)

        if thinking_enabled:
            return result
        return StructuredConversationResponse(
            response=result.response,
            thinking_summary=(),
            key_terms=result.key_terms,
        )

    def conversation_prompt(
        self,
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
        thinking_enabled: bool = False,
    ) -> str:
        if not recent_messages:
            raise StructuredGenerationError(
                "At least one conversation message is required."
            )
        return self._conversation_prompt(
            recent_messages,
            compacted_context=compacted_context,
            thinking_enabled=thinking_enabled,
        )

    def compact_context(
        self,
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
    ) -> str:
        if not recent_messages and not compacted_context:
            raise StructuredGenerationError(
                "Conversation context is required."
            )
        output = self.generator(
            self._compaction_prompt(recent_messages, compacted_context),
            COMPACTION_GENERATION_MAX_TOKENS,
        )
        return self._parse_compaction_output(output)

    def analyze_tree_update(
        self,
        question: str,
        response: str,
    ) -> GeneratedTreeUpdate:
        highlighted_terms = tuple(
            match.group(1)
            for match in re.finditer(r"\[([^\]]+)\]\(aq://[^)]+\)", response)
        )
        visible_response = self._visible_response_text(response)
        prompt = self._tree_analysis_prompt(
            question,
            visible_response,
            highlighted_terms,
        )
        first_output = self.generator(prompt, TREE_ANALYSIS_MAX_TOKENS)
        try:
            return self._parse_tree_update(first_output, visible_response)
        except StructuredGenerationError:
            repaired = self.generator(
                self._tree_analysis_repair_prompt(first_output, visible_response),
                TREE_ANALYSIS_MAX_TOKENS,
            )
            return self._parse_tree_update(repaired, visible_response)

    def label_tree_subject(self, insight_descriptions: Sequence[str]) -> GeneratedNodeSubject:
        cleaned = tuple(
            " ".join(description.split())
            for description in insight_descriptions
            if " ".join(description.split())
        )
        if not cleaned:
            raise StructuredGenerationError("At least one Insight is required.")
        prompt = self._node_subject_prompt(cleaned)
        first_output = self.generator(prompt, STRUCTURED_GENERATION_MAX_TOKENS)
        try:
            return self._parse_node_subject(first_output)
        except StructuredGenerationError:
            repaired = self.generator(
                self._node_subject_repair_prompt(first_output),
                STRUCTURED_GENERATION_MAX_TOKENS,
            )
            return self._parse_node_subject(repaired)

    @staticmethod
    def _visible_response_text(response: str) -> str:
        """Remove client-only aq:// annotations while preserving visible evidence."""
        return re.sub(
            r"\[([^\]]+)\]\(aq://[^)]+\)",
            lambda match: match.group(1),
            response,
        )

    def parse_conversation_output(
        self,
        output: str,
    ) -> StructuredConversationResponse:
        return self._parse_conversation_response(output)

    @staticmethod
    def _definition_prompt(
        term: str,
        source_excerpt: str,
        recent_messages: Sequence[ConversationMessage],
    ) -> str:
        context = "\n".join(
            f"{message.role}: {message.text}"
            for message in recent_messages[-12:]
        )
        return f"""
<TASK:CONTEXTUAL_DEFINITION>
Define the selected term exactly as it is being used in this conversation.
Be concise, accurate, and understandable to a thoughtful general reader.
Do not begin the definition with phrases like "In this context" or "In this conversation";
define the term directly.

Selected term: {term}
Source excerpt: {source_excerpt or "(not provided)"}
Recent conversation:
{context or "(not provided)"}

Return exactly one JSON object. Do not use Markdown or add commentary.
Use this exact shape:
{{
  "title": "{term}",
  "part_of_speech": "",
  "pronunciation": "",
  "definition": "A direct definition in 1-3 sentences.",
  "example": "One short example tied to this conversation."
}}
</TASK:CONTEXTUAL_DEFINITION>
""".strip()

    @staticmethod
    def _conversation_prompt(
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
        thinking_enabled: bool = False,
    ) -> str:
        context = "\n".join(
            f"{message.role}: {message.text}"
            for message in recent_messages[-12:]
        )
        thinking_instruction = (
            """
Also provide 2-4 short, user-facing bullets summarizing the approach you took. This summary
should explain the main considerations and structure, not reveal private scratch work or hidden
chain-of-thought.
""".strip()
            if thinking_enabled
            else """
Thinking mode is disabled. Do not generate an approach or reasoning summary. Return an empty
thinking_summary array.
""".strip()
        )
        thinking_example = (
            '"A short summary of one consideration or reasoning step."'
            if thinking_enabled
            else ""
        )
        carried_context = compacted_context.strip() if compacted_context else "(not provided)"
        return f"""
<TASK:CONVERSATION_RESPONSE>
Answer the user's latest inquiry as Aquinas, using the recent conversation for context.
Follow your normal warm, concise Thomistic style. Identify up to 8 important terms or short
phrases in your answer that would benefit from a contextual definition.
{thinking_instruction}
When it is relevant to the answer's structure, the response string may use simple Markdown:
# for a major heading, ## or ### for smaller headings, and *...* or **...** for brief emphasis.
Do not force headings into short answers.

Compacted context from earlier turns:
{carried_context}

Recent conversation since that checkpoint:
{context}

Return exactly one JSON object. Do not use Markdown outside JSON.
Use this exact shape and key order:
{{
  "thinking_summary": [
    {thinking_example}
  ],
  "response": "Your complete answer as one JSON string.",
  "key_terms": [
    {{
      "display_text": "Exact text copied from the response",
      "canonical_term": "Canonical name",
      "context_excerpt": "A short exact excerpt from the response containing display_text"
    }}
  ]
}}
Every display_text and context_excerpt must occur exactly in response. Use an empty key_terms
array when no term deserves highlighting.
</TASK:CONVERSATION_RESPONSE>
""".strip()

    @staticmethod
    def _compaction_prompt(
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None,
    ) -> str:
        context = "\n".join(
            f"{message.role}: {message.text}"
            for message in recent_messages[-20:]
        )
        return f"""
<TASK:COMPACT_CONVERSATION_CONTEXT>
Create a durable context checkpoint for a continuing conversation.
Preserve the user's questions, established conclusions, important distinctions, definitions,
open questions, preferences, quoted concepts, and any commitments needed to answer later turns.
Remove repetition, conversational filler, and obsolete wording. Do not answer the conversation
or address the user. Write a concise factual record for another model invocation.

Previous compacted checkpoint:
{compacted_context or "(not provided)"}

Turns since that checkpoint:
{context or "(not provided)"}

Return exactly one JSON object with no Markdown or commentary:
{{
  "summary": "The complete replacement context checkpoint."
}}
</TASK:COMPACT_CONVERSATION_CONTEXT>
""".strip()

    @staticmethod
    def _tree_analysis_prompt(
        question: str,
        response: str,
        highlighted_terms: Sequence[str] = (),
    ) -> str:
        highlighted = ", ".join(highlighted_terms) or "(none)"
        return f"""
<TASK:INSIGHT_TREE_UPDATE>
Name the broad subject of the user's question. This subject becomes the conversation's initial
Node Concept when the Tree is empty, so base it primarily on the question rather than on incidental
details in the answer. For a trivial exchange, return blank subject strings.

Then decide whether this turn introduces ONE genuinely important Node Concept seed worth
preserving in the Tree. A seed is a new central definition, distinction, causal relationship,
principle, or conclusion that changes or materially extends the inquiry. Most responses should
return no seed.
Explanatory detail, examples, applications, restatements, and merely useful facts are not seeds.
Highlighted terms are definition affordances, not automatic Insights or Node Concepts. Create a
Node Concept seed only when one is also the turn's central durable idea. Manual bookmarking
handles the rest as Insights.

Return either one node_seed object or null. Its evidence_excerpt must be copied exactly from the
answer.

Question:
{question}

Answer:
{response}

Highlighted terms in the answer:
{highlighted}

Return exactly one JSON object with no Markdown or commentary:
{{
  "subject": {{
    "label": "Two to five words",
    "summary": "One concise sentence."
  }},
  "node_seed": {{
    "label": "Two to eight words",
    "summary": "A self-contained contextual definition in one or two sentences.",
    "evidence_excerpt": "Exact text copied from the answer"
  }}
}}
Use null for node_seed unless this response contains a pivotal new Node Concept.
</TASK:INSIGHT_TREE_UPDATE>
""".strip()

    @staticmethod
    def _node_subject_prompt(insight_descriptions: Sequence[str]) -> str:
        insights = "\n".join(f"- {description}" for description in insight_descriptions)
        return f"""
<TASK:INSIGHT_TREE_NODE_SUBJECT>
Name the broad shared subject that best explains why these Insights belong together.
The Node label must be a two-to-five-word conceptual category, not a copy of an Insight title.

Insights:
{insights}

Return exactly one JSON object with no Markdown or commentary:
{{
  "label": "Two to five words",
  "summary": "One concise sentence describing the shared subject."
}}
</TASK:INSIGHT_TREE_NODE_SUBJECT>
""".strip()

    @staticmethod
    def _node_subject_repair_prompt(invalid_output: str) -> str:
        return f"""
<TASK:REPAIR_INSIGHT_TREE_NODE_SUBJECT>
Convert the previous output into one JSON object with non-empty label and summary strings.
The label must contain two to five words.

Previous output:
{invalid_output}
</TASK:REPAIR_INSIGHT_TREE_NODE_SUBJECT>
""".strip()

    @staticmethod
    def _tree_analysis_repair_prompt(invalid_output: str, response: str) -> str:
        return f"""
<TASK:REPAIR_INSIGHT_TREE_JSON>
Convert the previous output into one valid JSON object with keys subject and node_seed.
subject must contain label and summary. node_seed must be either null or one object with label,
summary, and evidence_excerpt. The evidence_excerpt must be copied exactly from the answer.
Use null for an ungrounded seed.

Answer:
{response}

Previous output:
{invalid_output}
</TASK:REPAIR_INSIGHT_TREE_JSON>
""".strip()

    @classmethod
    def _parse_compaction_output(cls, output: str) -> str:
        data = cls._extract_json_object(output)
        summary = cls._clean_string(data.get("summary"))
        if not summary:
            raise StructuredGenerationError("The model omitted the compacted context.")
        return summary

    @staticmethod
    def _conversation_repair_prompt(
        invalid_output: str,
        thinking_enabled: bool = False,
    ) -> str:
        thinking_instruction = (
            """
thinking_summary must contain 2-4 concise, user-facing strings that summarize the approach without
revealing private scratch work.
""".strip()
            if thinking_enabled
            else "Thinking mode is disabled, so thinking_summary must be an empty array."
        )
        return f"""
<TASK:REPAIR_CONVERSATION_JSON>
Convert the previous response into exactly one valid JSON object with no Markdown or commentary.
Preserve the substantive answer. Required keys are response, thinking_summary, and key_terms.
{thinking_instruction}
key_terms must be an array of objects with display_text,
canonical_term, and context_excerpt. Highlighted text and excerpts must be copied exactly from
response.

Previous response:
{invalid_output}
</TASK:REPAIR_CONVERSATION_JSON>
""".strip()

    @staticmethod
    def _repair_prompt(term: str, invalid_output: str) -> str:
        return f"""
<TASK:REPAIR_JSON>
The previous response was not valid contextual-definition JSON.
Convert it into exactly one JSON object with no Markdown or commentary.

Selected term: {term}
Previous response:
{invalid_output}

Required keys:
title, part_of_speech, pronunciation, definition, example
</TASK:REPAIR_JSON>
""".strip()

    @classmethod
    def _parse_definition(
        cls,
        output: str,
        fallback_title: str,
    ) -> ContextualDefinition:
        data = cls._extract_json_object(output)
        title = cls._clean_string(
            data.get("title") or data.get("word") or fallback_title
        )
        definition = cls._clean_string(
            data.get("definition") or data.get("meaning")
        )
        part_of_speech = cls._clean_string(
            data.get("part_of_speech"),
            required=False,
        )
        pronunciation = cls._clean_string(
            data.get("pronunciation"),
            required=False,
        )
        example = cls._clean_string(
            data.get("example"),
            required=False,
        )

        if not title:
            raise StructuredGenerationError("The model omitted the definition title.")
        if not definition:
            raise StructuredGenerationError("The model omitted the contextual definition.")

        return ContextualDefinition(
            title=title,
            part_of_speech=part_of_speech,
            pronunciation=pronunciation,
            definition=definition,
            example=example,
        )

    @classmethod
    def _parse_conversation_response(
        cls,
        output: str,
    ) -> StructuredConversationResponse:
        data = cls._extract_json_object(output)
        response = cls._clean_string(
            data.get("response") or data.get("answer")
        )
        if not response:
            raise StructuredGenerationError("The model omitted its response.")

        raw_summary = data.get("thinking_summary", [])
        if not isinstance(raw_summary, list):
            raise StructuredGenerationError("thinking_summary must be an array.")
        thinking_summary = tuple(
            summary
            for item in raw_summary[:4]
            if (summary := cls._clean_string(item, required=False))
        )

        raw_terms = data.get("key_terms", [])
        if not isinstance(raw_terms, list):
            raise StructuredGenerationError("key_terms must be an array.")

        terms: list[GeneratedKeyTerm] = []
        seen: set[str] = set()
        for raw_term in raw_terms[:8]:
            if not isinstance(raw_term, dict):
                continue
            requested_display = cls._clean_string(
                raw_term.get("display_text"),
                required=False,
            )
            display_text = cls._exact_substring(
                response,
                requested_display,
            )
            if not display_text:
                continue

            deduplication_key = display_text.casefold()
            if deduplication_key in seen:
                continue
            seen.add(deduplication_key)

            canonical_term = cls._clean_string(
                raw_term.get("canonical_term"),
                required=False,
            ) or display_text
            context_excerpt = cls._validated_excerpt(
                response=response,
                display_text=display_text,
                requested_excerpt=cls._clean_string(
                    raw_term.get("context_excerpt"),
                    required=False,
                ),
            )
            terms.append(
                GeneratedKeyTerm(
                    display_text=display_text,
                    canonical_term=canonical_term,
                    context_excerpt=context_excerpt,
                )
            )

        return StructuredConversationResponse(
            response=response,
            thinking_summary=thinking_summary,
            key_terms=tuple(terms),
        )

    @classmethod
    def _parse_tree_update(
        cls,
        output: str,
        response: str,
    ) -> GeneratedTreeUpdate:
        data = cls._extract_json_object(output)
        raw_subject = data.get("subject", {})
        if not isinstance(raw_subject, dict):
            raise StructuredGenerationError("subject must be an object.")
        label = cls._clean_string(raw_subject.get("label"), required=False)
        summary = cls._clean_string(raw_subject.get("summary"), required=False)
        if bool(label) != bool(summary):
            raise StructuredGenerationError(
                "subject label and summary must both be present or both be blank."
            )

        raw_seed = data.get("node_seed")
        if raw_seed is not None and not isinstance(raw_seed, dict):
            raise StructuredGenerationError("node_seed must be an object or null.")

        node_seed = None
        if isinstance(raw_seed, dict):
            seed_label = cls._canonical_tree_title(
                cls._clean_string(raw_seed.get("label"), required=False)
            )
            seed_summary = cls._clean_string(
                raw_seed.get("summary"),
                required=False,
            )
            requested_evidence = cls._clean_string(
                raw_seed.get("evidence_excerpt"),
                required=False,
            )
            evidence = cls._exact_substring(response, requested_evidence)
            if (
                seed_label
                and seed_summary
                and evidence
                and not seed_label.endswith("?")
                and not seed_summary.endswith("?")
                and not cls._is_non_durable_tree_candidate(
                    response=response,
                    evidence=evidence,
                )
                and len(seed_label.split()) <= 10
            ):
                node_seed = GeneratedTreeNodeSeed(
                    label=seed_label,
                    summary=seed_summary,
                    evidence_excerpt=evidence,
                )

        if node_seed and (not label or not summary):
            raise StructuredGenerationError(
                "A tree update with a Node Concept seed requires a subject label and summary."
            )
        return GeneratedTreeUpdate(
            subject_label=label,
            subject_summary=summary,
            node_seed=node_seed,
        )

    @staticmethod
    def _canonical_tree_title(title: str) -> str:
        lowered = title.casefold().rstrip(" .,:;!?")
        for suffix in (" defined", " definition"):
            if lowered.endswith(suffix):
                return title[: len(lowered) - len(suffix)].rstrip()
        return title

    @classmethod
    def _parse_node_subject(cls, output: str) -> GeneratedNodeSubject:
        data = cls._extract_json_object(output)
        label = cls._clean_string(data.get("label"), required=False)
        summary = cls._clean_string(data.get("summary"), required=False)
        if not label or not summary or not 2 <= len(label.split()) <= 5:
            raise StructuredGenerationError("The model returned an invalid Node subject.")
        return GeneratedNodeSubject(label=label, summary=summary)

    @staticmethod
    def _is_non_durable_tree_candidate(response: str, evidence: str) -> bool:
        normalized_response = " ".join(response.casefold().split()).strip(" .!")
        acknowledgement_phrases = {
            "done",
            "got it",
            "okay",
            "sounds good",
            "sure",
            "thank you",
            "thanks",
            "you're welcome",
        }
        if normalized_response in acknowledgement_phrases:
            return True

        normalized_evidence = " ".join(evidence.casefold().split())
        procedural_prefixes = (
            "click ",
            "open the ",
            "press ",
            "run the ",
            "select ",
            "tap ",
            "type ",
        )
        return normalized_evidence.startswith(procedural_prefixes)

    @staticmethod
    def _extract_json_object(output: str) -> dict:
        decoder = json.JSONDecoder()
        for index, character in enumerate(output):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(output[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise StructuredGenerationError("The model did not return a valid JSON object.")

    @staticmethod
    def _clean_string(value, required: bool = True) -> str:
        if value is None:
            if required:
                return ""
            return ""
        if not isinstance(value, str):
            raise StructuredGenerationError("Definition fields must be strings.")
        return " ".join(value.split())

    @staticmethod
    def _exact_substring(text: str, requested: str) -> str:
        if not requested:
            return ""
        start = text.casefold().find(requested.casefold())
        if start < 0:
            return ""
        return text[start:start + len(requested)]

    @classmethod
    def _validated_excerpt(
        cls,
        response: str,
        display_text: str,
        requested_excerpt: str,
    ) -> str:
        exact_excerpt = cls._exact_substring(response, requested_excerpt)
        if exact_excerpt and display_text.casefold() in exact_excerpt.casefold():
            return exact_excerpt

        start = response.find(display_text)
        lower = max(0, start - 80)
        upper = min(len(response), start + len(display_text) + 80)
        return response[lower:upper]
