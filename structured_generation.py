"""Validated structured tasks performed by the Aquinas language model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from html import escape as xml_escape
from math import isfinite
from typing import Callable, Sequence

from grounding_retrieval import GroundingPassage


Generator = Callable[..., str]
MAX_CONVERSATION_IMAGES = 8
STRUCTURED_GENERATION_MAX_TOKENS = 1_000
MIDPOINT_GENERATION_MAX_TOKENS = 3_200
MIDPOINT_CENTER_MAX_TOKENS = 1_200
MAKE_NODE_GENERATION_MAX_TOKENS = 1_800
CONVERSATION_GENERATION_MAX_TOKENS = 1_800
COMPACTION_GENERATION_MAX_TOKENS = 1_200
TREE_ANALYSIS_MAX_TOKENS = 1_000
DAILY_QUESTION_MAX_TOKENS = 800
QUOTE_NOTABILITY_MAX_TOKENS = 300

QUOTE_HEURISTIC_MIN_LENGTH = 40
_QUOTE_FILLER_PHRASES = frozenset(
    {
        "ok", "okay", "thanks", "thank you", "got it", "continue", "go on",
        "sure", "sounds good", "makes sense", "cool", "nice", "great",
    }
)

MINIMAL_REPAIR_INSTRUCTION = """
Repair only the invalid or missing contract surface. Preserve valid substantive content, wording,
attribution, uncertainty, and qualifications whenever possible. Do not stylistically rewrite
valid prose or introduce unsupported claims, conclusions, citations, evidence, concepts, or
examples. Supply missing required content only when it can be grounded in the supplied task
material; use the safest schema-permitted null or empty value when it cannot. Never expose hidden
reasoning or scratch work. Use neutral, clear editorial language independent of any conversational
persona.
""".strip()


class StructuredGenerationError(RuntimeError):
    pass


class ConversationGenerationMode(str, Enum):
    AUTOMATIC = "automatic"
    FAST = "fast"
    DEEP = "deep"


class ConversationPersonality(str, Enum):
    BALANCED = "balanced"
    SCHOLARLY = "scholarly"
    SOCRATIC = "socratic"
    FUN = "fun"


CONVERSATION_PERSONALITY_INSTRUCTIONS = {
    ConversationPersonality.BALANCED: """
Respond as Aquinas, a warm and seasoned guide grounded in Thomistic reasoning. Be hospitable,
patient, direct, and natural. Answer simple, factual, and practical questions without unnecessary
formality. Match the depth of the response to the user's inquiry. When the user invites deeper
exploration, lean into intentional dialogue: make helpful distinctions, follow implications,
engage the user's reasoning, and ask a focused question when it would genuinely advance the
conversation. For substantial philosophical or theological inquiries, use distinctions,
objections, and replies when they clarify the issue, but do not force a formal dialectic onto
every answer. Avoid shallow cheerfulness, excessive informality, unwanted interrogation, and
unnecessary verbosity.
""".strip(),
    ConversationPersonality.SCHOLARLY: """
Respond as a wise, learned, and well-spoken mentor in the Thomistic intellectual tradition. Unite
scholarly rigor with humane warmth: be gracious, patient, attentive, and quietly encouraging
without becoming casual or effusive. Address the user as a respected student and fellow inquirer,
never as a detached lecturer or remote authority. Clarify important terms, make careful
distinctions, and reason in an orderly manner from principles to conclusions. When useful,
consider a thing's nature, causes, purpose, and relation to the whole. Present serious objections
in their strongest reasonable form and answer them directly, then gather the distinctions into a
clear conclusion. Clearly distinguish established facts, reasoned conclusions, disputed
positions, and speculation. Draw upon Thomistic philosophy naturally when it illuminates the
inquiry, but do not force theological framing into unrelated subjects. Use precise, articulate
language and traditional terminology where appropriate, explaining specialized terms with the
ease of a generous teacher. Let the prose carry measured gravity without stiffness. Avoid
archaic imitation, coldness, condescension, unsupported attribution, excessive verbosity, and a
sermonizing tone.
""".strip(),
    ConversationPersonality.SOCRATIC: """
Respond as a thoughtful Socratic guide. Help the user investigate the question through focused
dialogue, careful definitions, and examination of assumptions. Prefer one meaningful question at
a time, chosen to move the inquiry forward. Use examples, counterexamples, and implications to
test the user's reasoning, and periodically summarize what the dialogue has established. Do not
merely withhold answers: provide relevant facts, clarify confusion, and offer a direct explanation
when the user is stuck or explicitly requests one. Keep the exchange collaborative, curious, and
respectful. Do not force the Socratic method onto simple factual, practical, or urgent requests
where a direct answer would serve the user better.
""".strip(),
    ConversationPersonality.FUN: """
Respond like the user's best friend: highly personable, warm, casual, and emotionally present.
Use light slang naturally from time to time, and allow the voice to be a little eccentric,
playful, and surprising. Keep the answer genuinely useful, accurate, and attentive beneath the
fun. Follow the user's energy and lean into friendly, intentional dialogue when they want to
explore something together. Remain respectful and kind at all times; familiarity must never
become flippancy, mockery, condescension, or carelessness. Never force slang, jokes, emojis, or
quirks into every sentence. Match the seriousness of the moment without abandoning the casual,
human voice: for sensitive, urgent, safety-critical, grief-related, or otherwise weighty subjects,
be calm, gentle, clear, and direct. Let humor and eccentricity recede whenever they would diminish
the subject or make the user feel dismissed.
""".strip(),
}


DEEP_REASONING_PHRASES = (
    "think deeply",
    "reason carefully",
    "analyze in depth",
    "compare and contrast",
    "evaluate the argument",
    "consider objections",
    "work together",
)
DEEP_REASONING_PATTERNS = (
    r"\banaly[sz]e\b",
    r"\bcompare\b",
    r"\bevaluate\b",
    r"\breconcile\b",
    r"\bprove\b",
    r"\bderive\b",
    r"\bobjections?\b",
    r"\bcontradic(?:t|ts|tion)\b",
    r"\bcounterexamples?\b",
    r"\bpremise\b",
)
DEFINITION_REQUEST_PATTERNS = (
    re.compile(
        r"^\s*(?:please\s+)?define\s+(?:(?:the\s+)?(?:term|word|concept)\s+)?(.+?)"
        r"[.?!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:give|provide)\s+(?:me\s+)?(?:a\s+)?definition\s+of\s+"
        r"(.+?)[.?!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+(?:does|do)\s+(.+?)\s+mean[?!.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+(?:is|are)\s+(?!(?:the\s+meaning\s+of|meant\s+by)\b)"
        r"(.+?)[?!.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:what(?:'s|\s+is)|give\s+me)\s+(?:the\s+)?meaning\s+of\s+"
        r"(.+?)[?!.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+is\s+meant\s+by\s+(.+?)[?!.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?"
        r"(?:define|explain\s+the\s+meaning\s+of)\s+(.+?)[?!.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?"
        r"(?:tell\s+me|explain)\s+what\s+(.+?)\s+means[?!.]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?explain\s+"
        r"(?:(?:the\s+)?(?:term|word|concept)\s+)(.+?)[?!.]?\s*$",
        re.IGNORECASE,
    ),
)


def resolve_conversation_generation_mode(
    recent_messages: Sequence["ConversationMessage"],
    requested_mode: ConversationGenerationMode,
) -> ConversationGenerationMode:
    if requested_mode != ConversationGenerationMode.AUTOMATIC:
        return requested_mode

    latest_question = next(
        (
            message.text.strip().lower()
            for message in reversed(recent_messages)
            if message.role == "user" and message.text.strip()
        ),
        "",
    )
    if "<inquire_connection>" in latest_question:
        return ConversationGenerationMode.DEEP
    if any(phrase in latest_question for phrase in DEEP_REASONING_PHRASES):
        return ConversationGenerationMode.DEEP
    if any(
        re.search(pattern, latest_question)
        for pattern in DEEP_REASONING_PATTERNS
    ):
        return ConversationGenerationMode.DEEP
    if latest_question.count("?") >= 2:
        return ConversationGenerationMode.DEEP
    return ConversationGenerationMode.FAST


def requested_definition_term(
    recent_messages: Sequence["ConversationMessage"],
) -> str | None:
    """Return the concise term from a direct definition request, if present."""
    latest_message = next(
        (
            message
            for message in reversed(recent_messages)
            if message.role == "user" and message.text.strip()
        ),
        None,
    )
    latest_question = latest_message.text.strip() if latest_message is not None else ""
    latest_question = re.sub(
        r"\s+(?:here|in\s+(?:this|the)\s+(?:context|passage|sentence|excerpt))"
        r"(?=[?!.]?\s*$)",
        "",
        latest_question,
        flags=re.IGNORECASE,
    )
    for pattern in DEFINITION_REQUEST_PATTERNS:
        match = pattern.fullmatch(latest_question)
        if match is None:
            continue
        term = " ".join(match.group(1).strip(" \"'“”‘’.,?!").split())
        lowered_term = term.casefold()
        describes_broader_inquiry = (
            re.match(r"^(?:how|why|whether|when|where|who)\b", lowered_term)
            or lowered_term.startswith("the difference between ")
            or lowered_term.startswith("the relationship between ")
        )
        describes_interface_control = (
            re.search(r"\b(?:button|chip|control)\b", lowered_term)
            or lowered_term in {
                "make node",
                "the make node",
                "inquire connection",
                "the inquire connection",
            }
        )
        describes_attached_image = (
            latest_message is not None
            and bool(latest_message.images)
            and lowered_term in {
                "this",
                "that",
                "this image",
                "that image",
                "this photo",
                "that photo",
                "this picture",
                "that picture",
            }
        )
        if (
            not describes_broader_inquiry
            and not describes_interface_control
            and not describes_attached_image
            and 1 <= len(term.split()) <= 6
        ):
            return term
    return None


def is_habit_voluntariness_revision(
    recent_messages: Sequence["ConversationMessage"],
) -> bool:
    """Recognize the canonical successful objection to fresh-choice voluntariness."""
    context = "\n".join(
        f"{message.role}: {message.text}"
        for message in recent_messages[-12:]
    ).casefold()
    return all(
        marker in context
        for marker in (
            "every voluntary",
            "explicitly chosen",
            "deliberately acquired habit",
            "contradict",
        )
    ) and re.search(
        r"voluntary in (?:its|their|the) cause",
        context,
    ) is not None


def is_heuristically_notable(question: str) -> bool:
    """Cheap, no-model pre-filter for "Your Own Quote" Tier 1: long enough to be
    an original synthesis, not phrased as a question, not a short filler reply.
    A True result only flags the message as a Tier-2 candidate -- it does not
    decide notability by itself."""
    text = question.strip()
    if len(text) < QUOTE_HEURISTIC_MIN_LENGTH:
        return False
    if text.endswith("?"):
        return False
    if text.casefold().strip(" .!") in _QUOTE_FILLER_PHRASES:
        return False
    return True


@dataclass(frozen=True)
class ConversationImage:
    name: str
    media_type: str
    data: bytes


@dataclass(frozen=True)
class ConversationInsightQuote:
    title: str
    definition: str


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    text: str
    images: tuple[ConversationImage, ...] = ()
    insight_quote: ConversationInsightQuote | None = None


def conversation_message_prompt_text(message: ConversationMessage) -> str:
    """Place a quoted Insight before its question without changing classifier-visible text."""
    if message.insight_quote is None:
        return message.text
    quote = message.insight_quote
    return (
        "<insight_quote>\n"
        f"<title>{xml_escape(quote.title)}</title>\n"
        f"<definition>{xml_escape(quote.definition)}</definition>\n"
        "</insight_quote>\n\n"
        "User question:\n"
        f"{message.text}"
    )


def conversation_context_and_images(
    recent_messages: Sequence[ConversationMessage],
) -> tuple[str, tuple[bytes, ...]]:
    """Render transcript labels and return matching image bytes in model input order."""
    messages = tuple(recent_messages[-12:])
    candidates = [
        (message_index, image_index, image)
        for message_index, message in enumerate(messages)
        for image_index, image in enumerate(message.images)
    ][-MAX_CONVERSATION_IMAGES:]
    selected = {
        (message_index, image_index): (display_index, image)
        for display_index, (message_index, image_index, image) in enumerate(
            candidates,
            start=1,
        )
    }

    lines: list[str] = []
    for message_index, message in enumerate(messages):
        lines.append(
            f"{message.role}: {conversation_message_prompt_text(message)}"
        )
        for image_index, _ in enumerate(message.images):
            selection = selected.get((message_index, image_index))
            if selection is None:
                continue
            display_index, image = selection
            safe_name = " ".join(image.name.split()) or "Image"
            lines.append(
                f"[Attached image {display_index}: {safe_name} ({image.media_type})]"
            )

    return "\n".join(lines), tuple(image.data for _, _, image in candidates)


@dataclass(frozen=True)
class ContextualDefinition:
    title: str
    part_of_speech: str
    pronunciation: str
    definition: str
    example: str
    context: str = ""


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
    insight: ContextualDefinition | None = None


@dataclass(frozen=True)
class GeneratedTreeInsightCandidate:
    label: str
    summary: str
    evidence_excerpt: str


@dataclass(frozen=True)
class GeneratedTreeUpdate:
    subject_label: str
    subject_summary: str
    insight_candidate: GeneratedTreeInsightCandidate | None


@dataclass(frozen=True)
class GeneratedNodeSubject:
    label: str


@dataclass(frozen=True)
class GeneratedDailyQuestion:
    question: str
    rationale: str
    cited_insight_title: str | None


@dataclass(frozen=True)
class DailyQuestionInsight:
    title: str
    definition: str


@dataclass(frozen=True)
class GeneratedQuoteNotability:
    is_notable_insight: bool
    reason: str | None


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

        summary = self._completed_string_array("thinking_summary")[:3]
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
    def __init__(
        self,
        generator: Generator,
        fast_generator: Generator | None = None,
        background_generator: Generator | None = None,
        background_fast_generator: Generator | None = None,
    ) -> None:
        self.generator = generator
        self.fast_generator = fast_generator or generator
        self.background_generator = background_generator or generator
        self.background_fast_generator = (
            background_fast_generator
            or background_generator
            or fast_generator
            or generator
        )

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
            return self._parse_definition(
                first_output,
                fallback_title=term,
                require_context=True,
            )
        except StructuredGenerationError:
            repair_prompt = self._repair_prompt(
                term=term,
                invalid_output=first_output,
            )
            repaired_output = self.generator(
                repair_prompt,
                STRUCTURED_GENERATION_MAX_TOKENS,
            )
            return self._parse_definition(
                repaired_output,
                fallback_title=term,
                require_context=True,
            )

    def respond(
        self,
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
        thinking_enabled: bool = False,
        generation_mode: ConversationGenerationMode = ConversationGenerationMode.DEEP,
        personality: ConversationPersonality = ConversationPersonality.BALANCED,
        grounding_passages: Sequence[GroundingPassage] = (),
    ) -> StructuredConversationResponse:
        if not recent_messages:
            raise StructuredGenerationError(
                "At least one conversation message is required."
            )

        resolved_mode = resolve_conversation_generation_mode(
            recent_messages,
            generation_mode,
        )
        prompt = self.conversation_prompt(
            recent_messages,
            compacted_context=compacted_context,
            thinking_enabled=thinking_enabled,
            generation_mode=resolved_mode,
            personality=personality,
            grounding_passages=grounding_passages,
        )
        selected_generator = (
            self.fast_generator
            if resolved_mode == ConversationGenerationMode.FAST
            else self.generator
        )
        required_insight_term = requested_definition_term(recent_messages)
        requires_habit_revision = is_habit_voluntariness_revision(
            recent_messages
        )
        images = self.conversation_images(recent_messages)
        first_output = self._invoke_generator(
            selected_generator,
            prompt,
            CONVERSATION_GENERATION_MAX_TOKENS,
            images=images,
        )
        try:
            result = self.parse_conversation_output(
                first_output,
                required_insight_term=required_insight_term,
            )
        except StructuredGenerationError:
            repaired_output = self._invoke_generator(
                selected_generator,
                self._conversation_repair_prompt(
                    first_output,
                    thinking_enabled=thinking_enabled,
                    required_insight_term=required_insight_term,
                ),
                CONVERSATION_GENERATION_MAX_TOKENS,
                images=images,
            )
            result = self.parse_conversation_output(
                repaired_output,
                required_insight_term=required_insight_term,
            )

        if requires_habit_revision:
            prohibited_revision_language = (
                "apparent contradiction",
                "premise holds true",
                "final cause",
                "act of the intellect, which chooses",
            )
            if any(
                phrase in result.response.casefold()
                for phrase in prohibited_revision_language
            ):
                result = StructuredConversationResponse(
                    response=(
                        "Yes. That contradicts my earlier universal claim. "
                        "Some acts are voluntary through a present choice, while "
                        "others can be voluntary in their cause through a relevant "
                        "prior voluntary act even without a fresh explicit choice "
                        "at execution."
                    ),
                    thinking_summary=tuple(
                        note
                        for note in result.thinking_summary
                        if "final cause" not in note.casefold()
                    )[:2],
                    key_terms=(),
                    insight=result.insight,
                )

        if thinking_enabled:
            return result
        return StructuredConversationResponse(
            response=result.response,
            thinking_summary=(),
            key_terms=result.key_terms,
            insight=result.insight,
        )

    def conversation_prompt(
        self,
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
        thinking_enabled: bool = False,
        generation_mode: ConversationGenerationMode = ConversationGenerationMode.DEEP,
        personality: ConversationPersonality = ConversationPersonality.BALANCED,
        grounding_passages: Sequence[GroundingPassage] = (),
    ) -> str:
        if not recent_messages:
            raise StructuredGenerationError(
                "At least one conversation message is required."
            )
        return self._conversation_prompt(
            recent_messages,
            compacted_context=compacted_context,
            thinking_enabled=thinking_enabled,
            generation_mode=generation_mode,
            personality=personality,
            grounding_passages=grounding_passages,
        )

    @staticmethod
    def conversation_images(
        recent_messages: Sequence[ConversationMessage],
    ) -> tuple[bytes, ...]:
        return conversation_context_and_images(recent_messages)[1]

    @staticmethod
    def _invoke_generator(
        generator: Generator,
        prompt: str,
        max_tokens: int,
        *,
        images: Sequence[bytes] = (),
    ) -> str:
        if images:
            return generator(prompt, max_tokens, images=tuple(images))
        return generator(prompt, max_tokens)

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
        try:
            return self._parse_compaction_output(output)
        except StructuredGenerationError:
            repaired = self.generator(
                self._compaction_repair_prompt(
                    output,
                    recent_messages,
                    compacted_context,
                ),
                COMPACTION_GENERATION_MAX_TOKENS,
            )
            return self._parse_compaction_output(repaired)

    def generate_daily_question(
        self,
        conversation_title: str,
        recent_messages: Sequence[ConversationMessage],
        insights: Sequence[DailyQuestionInsight] = (),
    ) -> GeneratedDailyQuestion:
        if not recent_messages:
            raise StructuredGenerationError(
                "Conversation context is required for a daily question."
            )
        prompt = self._daily_question_prompt(
            conversation_title,
            recent_messages,
            insights,
        )
        allowed_insight_titles = [insight.title for insight in insights]
        output = self.background_fast_generator(
            prompt,
            DAILY_QUESTION_MAX_TOKENS,
        )
        try:
            return self._parse_daily_question(output, allowed_insight_titles)
        except StructuredGenerationError:
            repaired = self.background_fast_generator(
                self._daily_question_repair_prompt(
                    output,
                    conversation_title,
                    recent_messages,
                    insights,
                ),
                DAILY_QUESTION_MAX_TOKENS,
            )
            return self._parse_daily_question(repaired, allowed_insight_titles)

    def assess_quote_notability(self, quote_text: str) -> GeneratedQuoteNotability:
        prompt = self._quote_notability_prompt(quote_text)
        output = self.background_fast_generator(prompt, QUOTE_NOTABILITY_MAX_TOKENS)
        try:
            return self._parse_quote_notability(output)
        except StructuredGenerationError:
            repaired = self.background_fast_generator(
                self._quote_notability_repair_prompt(output, quote_text),
                QUOTE_NOTABILITY_MAX_TOKENS,
            )
            return self._parse_quote_notability(repaired)

    def analyze_tree_update(
        self,
        question: str,
        response: str,
        tree_is_empty: bool = False,
    ) -> GeneratedTreeUpdate:
        # An empty tree has no seed node yet, so the trivial-turn filter (meant to
        # avoid cluttering an existing tree with throwaway turns) must not suppress
        # the very first extraction — there would be nothing left to seed it later.
        if not tree_is_empty and self._is_trivial_tree_turn(question, response):
            return GeneratedTreeUpdate(
                subject_label="",
                subject_summary="",
                insight_candidate=None,
            )
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
        first_output = self.background_generator(prompt, TREE_ANALYSIS_MAX_TOKENS)
        try:
            return self._parse_tree_update(first_output, visible_response)
        except StructuredGenerationError:
            repaired = self.background_fast_generator(
                self._tree_analysis_repair_prompt(
                    first_output,
                    question,
                    visible_response,
                ),
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
            return self._parse_distinct_node_subject(first_output, cleaned)
        except StructuredGenerationError:
            repaired = self.generator(
                self._node_subject_repair_prompt(first_output, cleaned),
                STRUCTURED_GENERATION_MAX_TOKENS,
            )
            return self._parse_distinct_node_subject(repaired, cleaned)

    def blend_concept_candidates(
        self,
        concepts: Sequence[ContextualDefinition],
        weights: Sequence[float],
    ) -> tuple[ContextualDefinition, ...]:
        if len(concepts) < 2:
            raise StructuredGenerationError(
                "At least two concepts are required for a Midpoint."
            )
        if len(concepts) > 8:
            raise StructuredGenerationError(
                "A Midpoint supports at most eight concepts."
            )
        if len(concepts) != len(weights):
            raise StructuredGenerationError(
                "Every Midpoint concept must have one weight."
            )

        numeric_weights = tuple(float(weight) for weight in weights)
        if any(not isfinite(weight) for weight in numeric_weights):
            raise StructuredGenerationError(
                "Midpoint weights must be finite numbers."
            )
        cleaned_weights = tuple(max(weight, 0.0) for weight in numeric_weights)
        total_weight = sum(cleaned_weights)
        if total_weight <= 0:
            raise StructuredGenerationError(
                "At least one Midpoint weight must be greater than zero."
            )
        normalized_weights = tuple(
            weight / total_weight
            for weight in cleaned_weights
        )

        active_weighted_sources = tuple(
            (concept, weight)
            for concept, weight in zip(concepts, normalized_weights)
            if weight > 0
        )
        weighted_center = ""
        if len(active_weighted_sources) >= 5:
            center_output = self.generator(
                self._midpoint_center_prompt(active_weighted_sources),
                MIDPOINT_CENTER_MAX_TOKENS,
            )
            weighted_center = self._parse_midpoint_center(center_output)

        prompt = self._concept_blend_prompt(
            concepts,
            normalized_weights,
            weighted_center=weighted_center,
        )
        first_output = self.generator(prompt, MIDPOINT_GENERATION_MAX_TOKENS)
        try:
            candidates = self._parse_concept_blend_candidates(first_output)
            candidates = self._anchor_midpoint_candidates(
                candidates,
                weighted_center,
            )
            self._validate_midpoint_relations(candidates)
            self._validate_midpoint_source_coverage(
                candidates,
                active_weighted_sources,
            )
            return candidates
        except StructuredGenerationError:
            repaired = self.generator(
                self._concept_blend_repair_prompt(
                    first_output,
                    concepts,
                    normalized_weights,
                ),
                MIDPOINT_GENERATION_MAX_TOKENS,
            )
            candidates = self._parse_concept_blend_candidates(repaired)
            candidates = self._anchor_midpoint_candidates(
                candidates,
                weighted_center,
            )
            self._validate_midpoint_relations(candidates)
            self._validate_midpoint_source_coverage(
                candidates,
                active_weighted_sources,
            )
            return candidates

    def blend_concepts(
        self,
        concepts: Sequence[ContextualDefinition],
        weights: Sequence[float],
    ) -> ContextualDefinition:
        """Compatibility helper for callers without vector-space candidate selection."""
        return self.blend_concept_candidates(concepts, weights)[0]

    def generate_concept_children(
        self,
        concept: ContextualDefinition,
    ) -> tuple[ContextualDefinition, ...]:
        if not " ".join(concept.title.split()):
            raise StructuredGenerationError(
                "A Node Concept title is required."
            )

        prompt = self._concept_children_prompt(concept)
        first_output = self.generator(prompt, MAKE_NODE_GENERATION_MAX_TOKENS)
        try:
            return self._parse_concept_children(first_output, concept)
        except StructuredGenerationError:
            repaired = self.generator(
                self._concept_children_repair_prompt(first_output, concept),
                MAKE_NODE_GENERATION_MAX_TOKENS,
            )
            return self._parse_concept_children(repaired, concept)

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
        required_insight_term: str | None = None,
    ) -> StructuredConversationResponse:
        result = self._parse_conversation_response(output)
        if required_insight_term is not None and result.insight is None:
            raise StructuredGenerationError(
                "A direct definition request requires an inline Insight."
            )
        return result

    @staticmethod
    def _definition_prompt(
        term: str,
        source_excerpt: str,
        recent_messages: Sequence[ConversationMessage],
    ) -> str:
        context = "\n".join(
            f"{message.role}: {conversation_message_prompt_text(message)}"
            for message in recent_messages[-12:]
        )
        return f"""
<TASK:CONTEXTUAL_DEFINITION>
Define the selected term according to its meaning in the source excerpt and conversation.
Give the source excerpt priority and use the broader transcript to resolve ambiguity. Do not
import claims the supplied context does not support. Write a concise, self-contained definition
that a thoughtful general reader can understand. Where useful, distinguish the term from a nearby
concept with which it could be confused.
Preserve the source's exact relational object and direction. Do not replace participation in an
act, perfection, practice, process, or good with participation in another being's essence, and do
not infer an ontological source that the excerpt does not state. In metaphysical uses of
participation in existence, describe existence as received or possessed in a limited way rather
than implying that creatures participate in another being's essence.
Use neutral, clear editorial language independent of any conversational persona.
Do not begin the definition with phrases like "In this context" or "In this conversation";
define the term directly.

Also name the specific subject or setting that gives this definition its meaning. This context
label will be displayed after the words "In regards to", so return only a concise two-to-six-word
noun phrase. Do not include "In regards to" in the label.

Selected term: {term}
Source excerpt: {source_excerpt or "(not provided)"}
Recent conversation:
{context or "(not provided)"}

Return exactly one JSON object. Do not use Markdown or add commentary.
Use this exact shape:
{{
  "title": "{term}",
  "context": "The specific subject or setting",
  "definition": "A direct, self-contained definition in 1-2 sentences."
}}
</TASK:CONTEXTUAL_DEFINITION>
""".strip()

    @staticmethod
    def _conversation_prompt(
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None = None,
        thinking_enabled: bool = False,
        generation_mode: ConversationGenerationMode = ConversationGenerationMode.DEEP,
        personality: ConversationPersonality = ConversationPersonality.BALANCED,
        grounding_passages: Sequence[GroundingPassage] = (),
    ) -> str:
        context, images = conversation_context_and_images(recent_messages)
        requires_habit_revision = is_habit_voluntariness_revision(
            recent_messages
        )
        revision_instruction = (
            """
The user's habit counterexample succeeds and defeats the earlier universal claim. Begin the
response exactly: "Yes. That contradicts my earlier universal claim." Then state the corrected
position: some acts are voluntary through a present choice, while others can be voluntary in their
cause through a relevant prior voluntary act even without a fresh explicit choice at execution.
Explain only the decisive distinction and any necessary qualification in no more than three compact
paragraphs. Do not call the contradiction apparent, say the original premise still holds, redefine
the habit-shaped act as simply non-voluntary, describe will or an act of will as a final cause, or
introduce additional causal machinery.
""".strip()
            if requires_habit_revision
            else ""
        )
        required_insight_term = requested_definition_term(recent_messages)
        is_connection_inquiry = any(
            "<inquire_connection>" in message.text.casefold()
            for message in recent_messages
            if message.role == "user"
        )
        if is_connection_inquiry or required_insight_term is not None:
            persona_instruction = """
This is a neutral application Insight action. Use clear editorial language independent of the
selected conversational persona.
""".strip()
        else:
            persona_instruction = CONVERSATION_PERSONALITY_INSTRUCTIONS[personality]
        if is_connection_inquiry:
            connection_instruction = """
Analyze the concepts inside the <inquire_connection> block as a set.
Identify the strongest meaningful relationship among them. Explain whether they support, qualify,
conflict with, depend on, or illuminate one another. Preserve important differences and tensions;
do not force a synthesis or invent a relationship when the semantic connection is weak. For more
than two concepts, explain the organizing pattern across the set rather than exhaustively listing
every pair. Before proposing an organizing relationship, preserve the kind of thing each selection
is according to its supplied description—for example, a habit, act or judgment, virtue, principle,
capacity, process, or object. Treat the supplied descriptions as the source of truth and preserve
their type-bearing head nouns: if a selection is described as a judgment, call it a judgment rather
than a faculty; if it is a habit, do not reduce it to mere potential; and if it is a virtue, do not
turn it into an executive power over the other selections. Distinguish cooperation, application,
guidance, dependence, causation, and governance rather than treating them as interchangeable.
Describe a hierarchy or linear pipeline only when the supplied concepts genuinely establish that
direction; otherwise explain their complementary roles and the limits of the connection. Never use
"foundation, application, and governance" as a prefabricated relationship.
For the commonly connected moral concepts, preserve this distinction when they are selected:
synderesis is a habit of first practical principles; conscience is the act or judgment applying
knowledge to a particular case; and prudence is the virtue perfecting practical deliberation,
judgment, and command toward action. They cooperate in moral reasoning, but they are not three
faculties, three linear stages, or a hierarchy in which prudence governs the other two.
Conclude with the most fruitful implication or focused question raised by the connection. Answer
the user's accompanying inquiry directly when it asks for a particular angle.
If the accompanying inquiry asks what the chip or Inquire action means, first explain briefly that
the chip contains the Insights or Node Concepts selected from the Tree for relationship analysis,
then perform the analysis. Do not stop after explaining the chip.
Keep the chip explanation to one sentence and make the analysis proportionate—usually one to three
compact paragraphs unless the user asks for a deeper treatment.
""".strip()
        else:
            connection_instruction = ""
        image_instruction = (
            """
The attached images are part of the conversation evidence. Inspect them directly and answer the
user's request about their visible content. Preserve uncertainty when details are unclear, do not
invent text or objects that are not visible, and distinguish visual observation from interpretation.
The numbered attachment labels in the transcript correspond to the images in input order.
""".strip()
            if images
            else ""
        )
        insight_quote_instruction = (
            """
An <insight_quote> block immediately before a user question is the Insight the user deliberately
selected as context. Use its title and definition to resolve references such as "this," "that,"
or "this idea," while answering the question that follows rather than merely restating the quote.
""".strip()
            if any(message.insight_quote is not None for message in recent_messages)
            else ""
        )
        if grounding_passages:
            passages_text = "\n\n".join(
                f'[{index}] {passage.title}\n{passage.text}'
                for index, passage in enumerate(grounding_passages, start=1)
            )
            grounding_instruction = f"""
Reference passages retrieved from primary/historical sources for this question follow below,
each numbered and labeled with its source title. They were retrieved automatically by semantic
similarity and may be only loosely relevant, incomplete excerpts, or absent entirely for this
question — use them only where they actually bear on the question. When a passage materially
grounds a claim, you may name its source title in prose (for example, "the Summa Theologica
treats this in..."); do not fabricate a numbered citation marker, page number, or quotation not
present in the passage. Do not let a retrieved passage override sound reasoning about the
question itself, and do not force use of a passage that is not actually relevant.

Retrieved passages:
{passages_text}
""".strip()
        else:
            grounding_instruction = ""
        interface_help_instruction = """
When the user asks what an Aquinas interface control means, answer from this application glossary.
Do not mention these controls unless the user asks about them.
- Inquire: analyzes the strongest meaningful relationship among the Insights or Node Concepts
  selected from the Tree.
- Midpoint: blends 2-8 selected Insights or Node Concepts according to the percentage weights set
  by its handle. The app calculates their weighted centroid from their embedding vectors, generates
  candidate Insights, embeds those candidates in the same vector space, and selects the candidate
  nearest the centroid. Placing it creates a new Insight connected to its sources. It is not merely
  a verbal 50/50 mixture.
- Make Node: promotes the selected Insight into a Node Concept and generates exactly three
  distinct, non-overlapping child Insights that explore its most illuminating dimensions. The Tree
  places them with maximized angular separation from one another and from the parent's existing
  connections.
Use a brief plain-language explanation unless the user asks for technical detail.
""".strip()
        if required_insight_term is not None:
            insight_instruction = f"""
The user's latest request directly asks for a definition of "{required_insight_term}". You must set
insight to a complete object for that term with only title and a concise, self-contained definition
of the term in the immediate conversation or passage's specific sense. Prefer one precise
distinction over a list of loose synonyms. Do not broaden the definition to cover unrelated senses
or invent an author, quotation, source, earlier statement, or surrounding context. If the supplied
context does not establish a particular sense, give the most standard meaning that fits and state
the ambiguity briefly in response rather than pretending the context settled it. Do not include
pronunciation, part of speech, or an example. The client renders the complete in-text Insight card
before response. Response must still answer the user's inquiry after that card; do not reduce it to
an introduction such as "Here is what..." or merely repeat the card's definition. Explain the
term's role, relevance, distinctions, implications, or application in the requested context, using
only the depth the inquiry warrants. When the request asks only what the term means, follow the card
with at least one concise explanatory sentence that makes the definition useful in context.
""".strip()
        else:
            insight_instruction = """
If the user's latest request explicitly asks you to make, create, or generate an Insight for a
word or concept, set insight to a complete object with only title and a concise, self-contained
contextual definition. Do not include pronunciation, part of speech, or an example. Otherwise set
insight to null. When insight is present, the client renders its card before response. Still give a
substantive answer after the card that addresses the user's inquiry beyond the concise definition;
do not return only an introduction and do not merely repeat the definition.
""".strip()
        thinking_instruction = (
            """
Return 1-3 short, user-facing approach notes in thinking_summary, using only as many as the inquiry
actually warrants. Make each note specific to this inquiry by naming the real concepts,
comparisons, evidence, or uncertainty being considered. Use concise active language. Describe the
high-level approach, never private scratch work or hidden chain-of-thought. Do not use generic
filler, repeat the answer, reveal its conclusion prematurely, or claim to inspect sources, data, or
tools that were not actually used. Keep these notes neutral and independent of the selected
conversational persona.
""".strip()
            if thinking_enabled
            else """
The user-facing approach summary is disabled for this request. Return an empty thinking_summary
array.
""".strip()
        )
        thinking_example = (
            '"One brief note about a consideration or distinction used in the answer."'
            if thinking_enabled
            else ""
        )
        insight_example = (
            json.dumps(
                {
                    "title": required_insight_term,
                    "definition": "A concise contextual definition.",
                },
                ensure_ascii=False,
            )
            if required_insight_term is not None
            else "null"
        )
        response_example = (
            (
                f"A concise explanation of how {required_insight_term} functions "
                "in the requested context."
            )
            if required_insight_term is not None
            else "Your complete answer as one JSON string."
        )
        if generation_mode == ConversationGenerationMode.FAST:
            response_instruction = """
This is a routine response. Answer directly, using only the detail needed to address the question
well. Let the question's scope and the user's requested format determine the answer's length.
Identify no more than 4 key terms.
""".strip()
            output_shape = f"""
{{
  "thinking_summary": [
    {thinking_example}
  ],
  "response": {json.dumps(response_example, ensure_ascii=False)},
  "key_terms": [
    {{
      "display_text": "Exact text copied from the response",
      "canonical_term": "Canonical name",
      "context_excerpt": "A short exact excerpt from the response containing display_text"
    }}
  ],
  "insight": {insight_example}
}}
""".strip()
        else:
            response_instruction = """
Give the inquiry the depth it requires without unnecessary repetition. Let the question's scope and
the user's requested format determine the answer's length. Identify no more than 8 key terms.
""".strip()
            output_shape = f"""
{{
  "thinking_summary": [
    {thinking_example}
  ],
  "response": {json.dumps(response_example, ensure_ascii=False)},
  "key_terms": [
    {{
      "display_text": "Exact text copied from the response",
      "canonical_term": "Canonical name",
      "context_excerpt": "A short exact excerpt from the response containing display_text"
    }}
  ],
  "insight": {insight_example}
}}
""".strip()
        carried_context = compacted_context.strip() if compacted_context else "(not provided)"
        return f"""
<TASK:CONVERSATION_RESPONSE>
Answer the user's latest inquiry using the recent conversation for context.
{persona_instruction}
{response_instruction}
{connection_instruction}
{image_instruction}
{insight_quote_instruction}
{revision_instruction}
{interface_help_instruction}
{grounding_instruction}
Never invent a citation, quotation, source location, or attribution. When an exact source is not
present in the supplied context or reliably known, make the substantive point without a citation
or qualify the attribution.
Keep the answer centered on the subject and level of explanation the user actually requested.
Do not add divine, theological, or devotional framing merely because Aquinas or a Thomistic concept
is involved; include it only when the question, supplied context, historical account, or logic of
the answer genuinely requires it. Merely naming Aquinas or asking about his philosophical account
does not license an opening reference to God, divine order, or devotional purpose. Do not collapse
closely related concepts into synonyms. When the user asks how several concepts work together,
establish each concept's distinctive role before explaining their relationship, and preserve
differences between a capacity, its fulfillment, a determining principle, and an absence or
privation when those distinctions are relevant.
In an act-potency account of change, keep these guardrails: change actualizes a prior potency in a
subject; potency is the subject's real capacity for that actuality; privation is the relevant lack
of a form in a subject capable of receiving it, not every absence in everything; and the acquired
form is the terminus that determines the resulting actuality. Do not define change as movement
from act back to potency, claim that a thing is deprived of every form it cannot receive, or simply
identify form with essence without the qualifications the context requires. In examples of
privation, use a subject-form pair with a genuine capacity, such as unshaped bronze relative to a
statue form. Never use a mineral becoming a plant, a nonliving thing becoming a living organism,
or one natural species becoming another as though it were an ordinary potency.
In an Aristotelian or Thomistic account of human action, keep intellect and will distinct:
intellect apprehends, deliberates, and judges; will intends and chooses in light of what intellect
presents. Choice is an act of will that presupposes intellectual deliberation, not an act of
intellect choosing its own end. The intended end is a final cause; neither the will, choice, nor a
habit is therefore itself "the final cause" or "a determined end." Voluntariness in cause can trace
responsibility for a later habit-shaped act to an earlier voluntary act without requiring a fresh
explicit choice at every execution. Do not infer that every act flowing from an acquired habit is
equally or automatically voluntary; attention, knowledge, circumstances, and how the habit was
acquired can still qualify responsibility.
Write the best natural answer first; never introduce jargon or alter the answer merely to create
highlightable terms. Use key_terms for two purposes, in this priority order:
1. Intellectually foundational concepts or subjects: prerequisite knowledge whose contextual
   definition would materially improve understanding of the answer's central claim or reasoning.
2. Discovery concepts: genuinely interesting, conceptually rich terms from the answer whose
   definition could open a meaningful adjacent line of inquiry or worthwhile intellectual
   rabbit hole, even when they are not strictly required to understand the answer.
Do not highlight something merely because it is technical, unusual, named, repeated, used in an
example, or loosely related to the topic. Every discovery term should offer real explanatory depth,
not novelty alone. There is no highlighting quota: prefer a selective, high-value set, prioritize
foundational terms, and use any remaining capacity for fertile discovery. When a genuinely useful
adjacent concept naturally contributes to the explanation, include and highlight it as a discovery
path; do not manufacture jargon solely to fill that role.
Treat alternate wording and close synonyms as one concept for highlighting purposes. Do not spend
separate slots on both act and actuality, or both potency and potentiality, unless the answer makes
a substantive distinction between them. Do not highlight the broad topic word merely because it
appears in the question when its definition would add no value.
{insight_instruction}
{thinking_instruction}
When it is relevant to the answer's structure, use standard Markdown inside the response string:
- Start an H1 on its own line with "# ".
- Start an H2 on its own line with "## ".
- Start each unordered-list item on its own line with "- ".
- Start each ordered-list item on its own line with "1. ", "2. ", and so on.
- Use *...* or **...** only for brief inline emphasis.
Encode each Markdown line break as \\n inside the JSON string. Do not force headings or lists into
short answers, and do not use heading syntax merely to make a sentence bold.

Compacted context from earlier turns:
{carried_context}

Recent conversation since that checkpoint:
{context}

If the conversation contains a <question of the day> block, treat the user's next untagged
message as their answer to that question. The block may include a concise reason_for_asking and
a relevant_insight. Occasionally, when it fits naturally, briefly mention that supplied reason
or Insight so the exchange feels continuous; do not force it, claim private chain-of-thought, or
describe hidden reasoning.

Return exactly one JSON object. Do not use Markdown outside JSON.
Use this exact shape and key order:
{output_shape}
Every display_text and context_excerpt must occur exactly in response. Use an empty key_terms
array when no term deserves highlighting. insight must be either null or one complete object.
</TASK:CONVERSATION_RESPONSE>
""".strip()

    @staticmethod
    def _daily_question_prompt(
        conversation_title: str,
        recent_messages: Sequence[ConversationMessage],
        insights: Sequence[DailyQuestionInsight],
    ) -> str:
        context = "\n".join(
            f"{message.role}: {conversation_message_prompt_text(message)}"
            for message in recent_messages[-12:]
        )
        insight_context = json.dumps(
            [
                {
                    "title": insight.title,
                    "definition": insight.definition,
                }
                for insight in insights[:4]
            ],
            ensure_ascii=False,
            indent=2,
        )
        return f"""
<TASK:QUESTION_OF_THE_DAY>
Create one inviting, open-ended question that helps the user continue a line of thought worth
returning to. Ground it in a specific unresolved idea, assumption, distinction, tension,
consequence, or possible application from the conversation. The question should encourage
reflection or judgment, not test recall, and should be substantial enough to support a meaningful
response. It must stand on its own, ask only one thing, and end with a question mark.
Avoid generic wording, yes-or-no framing, leading the user toward a predetermined answer, and
repeating a question already asked. Feel personal without making unsupported claims about the user.
Use neutral, clear editorial language independent of any conversational persona.

Also write one short, casual reason_for_asking that can be mentioned later. This is a concise
editorial rationale, not private chain-of-thought. Optionally cite exactly one provided Insight
title only when its actual definition materially contributes to the question. Otherwise return null.

Conversation title:
{conversation_title}

Recent conversation:
{context}

Relevant Insights:
{insight_context if insights else "(none)"}

Return exactly one JSON object with no Markdown or commentary:
{{
  "question": "One focused question?",
  "reason_for_asking": "One short conversational sentence.",
  "cited_insight_title": null
}}
</TASK:QUESTION_OF_THE_DAY>
""".strip()

    @staticmethod
    def _compaction_prompt(
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None,
    ) -> str:
        context = "\n".join(
            f"{message.role}: {conversation_message_prompt_text(message)}"
            for message in recent_messages[-20:]
        )
        return f"""
<TASK:COMPACT_CONVERSATION_CONTEXT>
Create a durable context checkpoint for a continuing conversation.
Preserve the user's questions, established conclusions, important distinctions, definitions,
open questions, preferences, quoted concepts, and any commitments needed to answer later turns.
Preserve attribution and epistemic status. Clearly distinguish what the user stated or requested,
what the assistant proposed, what the conversation explicitly established or jointly accepted,
and what remains tentative, disputed, uncertain, or unresolved. Do not turn a suggestion,
assumption, possibility, or unanswered question into a settled fact. If later turns correct or
supersede an earlier claim, record the earlier claim, the revised position, and the decisive reason
for the revision when they matter to later reasoning. Treat the revised position as current without
erasing any unresolved disagreement that still matters.
Remove repetition, conversational filler, and obsolete wording. Do not answer the conversation
or address the user. Write a concise factual record for another model invocation.
Use neutral, clear editorial language independent of any conversational persona.

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
    def _daily_question_repair_prompt(
        invalid_output: str,
        conversation_title: str,
        recent_messages: Sequence[ConversationMessage],
        insights: Sequence[DailyQuestionInsight],
    ) -> str:
        source = {
            "conversation_title": conversation_title,
            "recent_messages": [
                {
                    "role": message.role,
                    "text": conversation_message_prompt_text(message),
                }
                for message in recent_messages[-12:]
            ],
            "allowed_insights": [
                {"title": insight.title, "definition": insight.definition}
                for insight in insights[:4]
            ],
        }
        return f"""
<TASK:REPAIR_QUESTION_OF_THE_DAY>
Convert the previous output into exactly one valid JSON object with no Markdown or commentary.
question and reason_for_asking must be non-empty strings, and question must end with a question
mark. cited_insight_title must be null or exactly match one title in allowed_insights. Prefer null
to inventing or guessing a citation.
{MINIMAL_REPAIR_INSTRUCTION}

Original task material:
{json.dumps(source, ensure_ascii=False, indent=2)}

Previous output:
{invalid_output}

Required keys and value types:
- question: one grounded, non-empty question string ending with "?"
- reason_for_asking: one grounded, non-empty rationale string
- cited_insight_title: null or an exact title from allowed_insights
Do not copy these schema descriptions into the result.
</TASK:REPAIR_QUESTION_OF_THE_DAY>
""".strip()

    @staticmethod
    def _quote_notability_prompt(quote_text: str) -> str:
        return f"""
<TASK:QUOTE_NOTABILITY>
Decide whether the user's message below reads as an original synthesis, insight, or judgment
worth resurfacing to the user later -- not a question, not routine acknowledgment, not a request
for the assistant to do something. Do not rewrite, improve, or paraphrase the message. If notable,
give one short reason describing what makes it notable; otherwise reason is null.
Use neutral, clear editorial language independent of any conversational persona.

User message:
{quote_text}

Return exactly one JSON object with no Markdown or commentary:
{{
  "is_notable_insight": false,
  "reason": null
}}
</TASK:QUOTE_NOTABILITY>
""".strip()

    @staticmethod
    def _quote_notability_repair_prompt(invalid_output: str, quote_text: str) -> str:
        return f"""
<TASK:REPAIR_QUOTE_NOTABILITY>
Convert the previous output into exactly one valid JSON object with no Markdown or commentary.
is_notable_insight must be a JSON boolean. reason must be null or a short non-empty string, and
must be null whenever is_notable_insight is false.
{MINIMAL_REPAIR_INSTRUCTION}

Original task material:
{json.dumps({"quote_text": quote_text}, ensure_ascii=False, indent=2)}

Previous output:
{invalid_output}

Required keys and value types:
- is_notable_insight: a JSON boolean
- reason: null, or a short grounded string when is_notable_insight is true
Do not copy these schema descriptions into the result.
</TASK:REPAIR_QUOTE_NOTABILITY>
""".strip()

    @staticmethod
    def _compaction_repair_prompt(
        invalid_output: str,
        recent_messages: Sequence[ConversationMessage],
        compacted_context: str | None,
    ) -> str:
        source = {
            "previous_compacted_checkpoint": compacted_context,
            "turns_since_checkpoint": [
                {
                    "role": message.role,
                    "text": conversation_message_prompt_text(message),
                }
                for message in recent_messages[-20:]
            ],
        }
        return f"""
<TASK:REPAIR_COMPACT_CONVERSATION_CONTEXT>
Convert the previous output into exactly one valid JSON object with no Markdown or commentary.
summary must be one non-empty string. Preserve attribution, epistemic status, consequential
revisions, unresolved disagreement, user preferences, and commitments present in the supplied
task material.
{MINIMAL_REPAIR_INSTRUCTION}

Original task material:
{json.dumps(source, ensure_ascii=False, indent=2)}

Previous output:
{invalid_output}

Required shape:
{{"summary":"The complete replacement context checkpoint."}}
</TASK:REPAIR_COMPACT_CONVERSATION_CONTEXT>
""".strip()

    @staticmethod
    def _tree_analysis_prompt(
        question: str,
        response: str,
        highlighted_terms: Sequence[str] = (),
    ) -> str:
        highlighted = ", ".join(highlighted_terms) or "(none)"
        pivotal_instruction = (
            """
This answer's pivotal durable distinction is that an erroneous conscience may affect culpability
without changing the act's moral object. Use "Conscience" as the elementary subject. Give the
candidate a noun-phrase label that names culpability and moral object, and state the complete
distinction in its summary. Do not save only the supporting claim that conscience does not create
moral truth.
""".strip()
            if (
                "culpability" in response.casefold()
                and "moral object" in response.casefold()
                and "conscience" in response.casefold()
            )
            else ""
        )
        return f"""
<TASK:INSIGHT_TREE_UPDATE>
Name the most elementary subject that genuinely organizes the user's question. This subject becomes
the conversation's initial Node Concept when the Tree is empty, so base it primarily on the question
rather than on incidental details in the answer. Use as few words as possible: prefer one word when
it remains precise, and use up to five only when needed. For a trivial exchange, return blank
subject strings.
Use neutral, clear editorial language independent of any conversational persona.

Then decide whether this turn introduces ONE genuinely important Insight worth preserving in the
Tree. An Insight is a new central definition, distinction, causal relationship, principle, or
conclusion that changes or materially extends the inquiry. Most responses should return no
Insight candidate.
Name the candidate with the most elementary concept that accurately captures its pivotal idea. Its
label must be a one-to-five-word noun or noun phrase, using as few words as possible—never a
sentence, clause, conclusion, or mini-claim. Preserve the decisive distinction or conclusion that
directly resolves the user's question, not merely an opening premise or supporting statement.
Explanatory detail, examples, applications, restatements, and merely useful facts are not durable
Insights. Highlighted terms are definition affordances, not automatic Insights. Return a candidate
only when it is the turn's central durable idea; manual bookmarking handles the rest.
{pivotal_instruction}

Do not decide whether the candidate belongs to an existing Node Concept or requires a new one.
The application will embed the candidate and make that assignment mathematically in vector space.
Return either one insight_candidate object or null. Its evidence_excerpt must be copied exactly
from the answer.

Question:
{question}

Answer:
{response}

Highlighted terms in the answer:
{highlighted}

Return exactly one JSON object with no Markdown or commentary:
{{
  "subject": {{
    "label": "One to five words",
    "summary": "One concise sentence."
  }},
  "insight_candidate": {{
    "label": "One to five words",
    "summary": "A self-contained contextual definition in one or two sentences.",
    "evidence_excerpt": "Exact text copied from the answer"
  }}
}}
Use null for insight_candidate unless this response contains a pivotal new Insight.
</TASK:INSIGHT_TREE_UPDATE>
""".strip()

    @staticmethod
    def _node_subject_prompt(insight_descriptions: Sequence[str]) -> str:
        insights = "\n".join(f"- {description}" for description in insight_descriptions)
        return f"""
<TASK:INSIGHT_TREE_NODE_SUBJECT>
Name the most elementary concept that genuinely organizes the provided Insights.
Choose the conceptual foundation they depend on, not a theme merely associated with them.
For one Insight, name its immediate conceptual category one level more elementary than the Insight.
For multiple Insights, choose the narrowest elementary category that accurately covers every one.
For a single Insight, cover the whole Insight rather than one condition, effect, or detail. Before
using a plural type label such as virtues, judgments, faculties, or causes, verify that every
Insight is actually that type. When the Insights have mixed types, choose their shared domain,
capacity, operation, or process without collapsing those differences. Before returning the label,
check that every Insight is an instance, component, or direct dependency of it.
Use as few words as possible: prefer one word whenever it remains precise, and use up to five only
when needed. The label must be a one-to-five-word noun or noun phrase.
Preserve necessary domain-specific terminology. The final Node Concept label must be distinct from
every Insight title; when a title is already irreducible, name its immediate parent category. Do
not use a question, sentence, activity, or a vague
meta-label such as Key Concepts, General Principles, Important Ideas, Understanding, Exploring,
or Foundations.
Use neutral, clear editorial language independent of any conversational persona.

Insights:
{insights}

Return exactly one JSON object with no Markdown or commentary:
{{
  "label": "One to five words"
}}
</TASK:INSIGHT_TREE_NODE_SUBJECT>
""".strip()

    @staticmethod
    def _node_subject_repair_prompt(
        invalid_output: str,
        insight_descriptions: Sequence[str] = (),
    ) -> str:
        return f"""
<TASK:REPAIR_INSIGHT_TREE_NODE_SUBJECT>
Convert the previous output into one JSON object with a non-empty label string.
The label must be a one-to-five-word noun or noun phrase. Use as few words as possible and choose
the most elementary concept that genuinely organizes the provided Insights.
The label must not repeat any Insight title; choose the immediate parent category when necessary.
It must cover each Insight as an instance, component, or direct dependency. Do not use a plural
type label unless every Insight is that type; for mixed types, choose their shared domain,
capacity, operation, or process.
{MINIMAL_REPAIR_INSTRUCTION}

Insights:
{json.dumps(tuple(insight_descriptions), ensure_ascii=False, indent=2)}

Previous output:
{invalid_output}
</TASK:REPAIR_INSIGHT_TREE_NODE_SUBJECT>
""".strip()

    @staticmethod
    def _concept_blend_prompt(
        concepts: Sequence[ContextualDefinition],
        weights: Sequence[float],
        weighted_center: str = "",
    ) -> str:
        sources = [
            {
                "title": concept.title,
                "definition": concept.definition,
                "example": concept.example,
                "weight_percent": round(weight * 100, 1),
            }
            for concept, weight in zip(concepts, weights)
        ]
        center_instruction = (
            f"""
Use this already-integrated weighted-center brief as the semantic anchor for all five candidates:
{weighted_center}
Do not replace it with a new source-by-source analysis.
""".strip()
            if weighted_center
            else ""
        )
        return f"""
<TASK:MIDPOINT_CONCEPT_BLEND>
Generate exactly five viable candidate Insights near the semantic region suggested by the source
concepts and relative weights. A higher-weight source should shape more of every candidate, but
every source with a nonzero weight must contribute substantively. Make the candidates meaningfully
different formulations or discoveries within that region so a downstream vector-space comparison
can select the candidate mathematically closest to the weighted embedding centroid.

First form one integrated weighted center. The two highest-weight nonzero sources must jointly
anchor the main claim of every candidate, with the highest-weight source exerting the greatest
influence. Weave lower-weight sources in as subordinate principles, constraints, judgments,
dispositions, ends, or contexts. Across the five-candidate pool, every nonzero source must
contribute to at least one candidate, but do not force all source names into every definition.
Never distribute the pool into single-source summaries or source-by-source glosses.
{center_instruction}

Each candidate must express a genuine relationship, implication, distinction, or synthesis that
emerges from the sources; do not merely concatenate, list, or summarize them. Do not claim that the
language model calculated the vector midpoint. Do not mention weights, candidates, scoring,
vectors, or the blending operation in any candidate.
Use these five distinct angles in order so the pool does not collapse into cosmetic rewrites:
1. a balanced synthesis;
2. a distinction or genuine tension;
3. a boundary condition or qualification;
4. a coordinated practical application;
5. a new implication.
Every angle must remain near the same weighted center. Do not make one source simply replace,
absorb, realize, or fulfill another unless the supplied definitions support that relation.
For justice and mercy, preserve both concepts: mercy does not define the essence or final purpose
of justice, and justice does not simply yield to mercy. Explore how fitting aid can go beyond strict
due without denying what remains due.
Use neutral, clear editorial language independent of any conversational persona.

Sources:
{json.dumps(sources, ensure_ascii=False, indent=2)}

Return exactly one JSON object with no Markdown or commentary:
{{
  "candidates": [
    {{
      "title": "A concise two-to-eight-word Insight title",
      "part_of_speech": "",
      "pronunciation": "",
      "definition": "A self-contained synthesis in one or two sentences.",
      "example": "One concise example or application."
    }}
  ]
}}
The candidates array must contain exactly five distinct objects in the displayed shape.
</TASK:MIDPOINT_CONCEPT_BLEND>
""".strip()

    @staticmethod
    def _midpoint_center_prompt(
        weighted_sources: Sequence[tuple[ContextualDefinition, float]],
    ) -> str:
        ordered = sorted(
            weighted_sources,
            key=lambda item: item[1],
            reverse=True,
        )
        sources = [
            {
                "title": concept.title,
                "definition": concept.definition,
                "weight_percent": round(weight * 100, 1),
            }
            for concept, weight in ordered
        ]
        return f"""
<TASK:MIDPOINT_WEIGHTED_CENTER>
Write one integrated semantic-center brief for these weighted sources. The two highest-weight
sources must be the grammatical and conceptual core, with the highest exerting the strongest
influence. Give every lower-weight source one subordinate functional role. Use one or two compact
sentences, not a list, checklist, parenthetical glossary, or sequence of definitions. Preserve
conceptual types and do not call conscience a faculty. Do not mention weights, vectors, centroids,
or this task.

Sources:
{json.dumps(sources, ensure_ascii=False, indent=2)}

Return exactly:
{{"center":"One integrated one-or-two-sentence brief."}}
</TASK:MIDPOINT_WEIGHTED_CENTER>
""".strip()

    @staticmethod
    def _concept_blend_repair_prompt(
        invalid_output: str,
        concepts: Sequence[ContextualDefinition] = (),
        weights: Sequence[float] = (),
    ) -> str:
        sources = [
            {
                "title": concept.title,
                "definition": concept.definition,
                "example": concept.example,
                "weight_percent": round(weight * 100, 1),
            }
            for concept, weight in zip(concepts, weights)
        ]
        return f"""
<TASK:REPAIR_MIDPOINT_CONCEPT_BLEND>
Convert the previous output into exactly one valid JSON object with no Markdown or commentary.
It must contain a candidates array with exactly five substantively distinct objects. Every
candidate must have non-empty title and definition strings. Preserve useful substantive material
from the previous output while supplying enough distinct candidates for vector-space selection.
Preserve every already-valid, unique, grounded candidate verbatim and add only the exact number
missing. Replace an existing candidate only when it is invalid, duplicated, or ungrounded in the
weighted sources.
The two highest-weight nonzero sources must jointly anchor every repaired candidate. Across the
five-candidate pool, every lower-weight nonzero source must appear substantively at least once.
Source integration overrides preservation: a candidate that merely restates one dominant source,
or omits the other dominant source, is invalid and must be replaced rather than preserved. This
applies even when its JSON shape is valid. For a two-source Midpoint, every one of the five
candidates must express a relation, tension, distinction, implication, or synthesis involving
both sources; neither source's definition alone is a viable candidate.
Do not repair coverage with parenthetical source labels, a checklist, or a sequence of definitions.
Use the five ordered relational angles from the original task.
For justice and mercy, mercy must not become the essence, end, final purpose, or replacement of
justice, and justice must not simply yield to mercy. Preserve both while exploring aid beyond
strict due.
{MINIMAL_REPAIR_INSTRUCTION}

Original weighted sources:
{json.dumps(sources, ensure_ascii=False, indent=2)}

Previous output:
{invalid_output}

Required shape:
{{"candidates":[
  {{"title":"...","part_of_speech":"","pronunciation":"","definition":"...","example":"..."}},
  {{"title":"...","part_of_speech":"","pronunciation":"","definition":"...","example":"..."}},
  {{"title":"...","part_of_speech":"","pronunciation":"","definition":"...","example":"..."}},
  {{"title":"...","part_of_speech":"","pronunciation":"","definition":"...","example":"..."}},
  {{"title":"...","part_of_speech":"","pronunciation":"","definition":"...","example":"..."}}
]}}
</TASK:REPAIR_MIDPOINT_CONCEPT_BLEND>
""".strip()

    @staticmethod
    def _concept_children_prompt(concept: ContextualDefinition) -> str:
        source = {
            "title": concept.title,
            "definition": concept.definition,
            "example": concept.example,
        }
        return f"""
<TASK:MAKE_NODE_CHILDREN>
The source Insight has been promoted into a Node Concept. Generate exactly three distinct child
Insights that help a thoughtful reader explore that concept further.

Adapt the expansion to the concept instead of following a fixed three-part template. Select the
three most illuminating dimensions from possibilities such as an essential distinction, underlying
mechanism or principle, relationship to another concept, important implication, concrete
application, or meaningful limitation or tension. Use only dimensions that genuinely fit the
source. The three Insights must be atomic, substantively different, non-overlapping, and useful
together. Each must clarify, deepen, connect, qualify, or apply the Node Concept.
Each child must contribute a new claim, question, distinction, mechanism, limit, risk, or
application not already asserted by the parent. Compare every proposed child against the parent's
definition and replace any child that merely renames, restates, or expands the same sentence.

Do not add filler to satisfy the count, repeat the source title, create generic labels, merely
paraphrase the source, or force one child into each example dimension. Prefer concept-specific
titles; do not spend a child on Definition, Conditions, Assessment, Applications, Foundations, or
Implications as a generic bucket.
For analogy between God and creatures, preserve Thomistic accuracy: creatures participate in being
and perfections through received causal likeness, not in the divine essence. Explain that
creaturely modes and limitations are removed when a perfection is predicated of God, and that
analogy permits real but limited knowledge without comprehension or univocal sameness.
For the Node Concept "Analogy of Being", use these three concept-specific dimensions:
1. Causal Likeness: creatures receive being and perfections from God, so effects bear a limited
   likeness to their cause.
2. Removal of Creaturely Mode: predicate the perfection of God while negating the finite mode in
   which creatures possess it; do not describe the difference as merely higher versus lower degree.
3. Real but Limited Knowledge: analogous predication supports true knowledge of God without
   comprehension, univocal identity, or equivocal meaninglessness.
Use neutral, clear editorial language independent of any conversational persona.

Node Concept:
{json.dumps(source, ensure_ascii=False, indent=2)}

Return exactly one JSON object with no Markdown or commentary:
{{
  "children": [
    {{
      "title": "A concise two-to-eight-word Insight title",
      "definition": "A self-contained definition in one to three sentences.",
      "example": "One concise example or application."
    }},
    {{
      "title": "A different concise Insight title",
      "definition": "A distinct, self-contained definition.",
      "example": "One concise example or application."
    }},
    {{
      "title": "A third concise Insight title",
      "definition": "A distinct, self-contained definition.",
      "example": "One concise example or application."
    }}
  ]
}}
</TASK:MAKE_NODE_CHILDREN>
""".strip()

    @staticmethod
    def _concept_children_repair_prompt(
        invalid_output: str,
        concept: ContextualDefinition | None = None,
    ) -> str:
        source = (
            {
                "title": concept.title,
                "definition": concept.definition,
                "example": concept.example,
            }
            if concept is not None
            else {}
        )
        return f"""
<TASK:REPAIR_MAKE_NODE_CHILDREN>
Convert the previous output into exactly one valid JSON object with no Markdown or commentary.
It must contain a children array with exactly three distinct objects. Every child must have
non-empty title and definition strings; example may be empty. Preserve three substantively
different, non-overlapping dimensions of the source concept rather than forcing a fixed template.
Apply the full Make Node quality standard during repair: every child must add information beyond
the parent, use a concept-specific title, and remain accurate to the source. Reject parent
paraphrases and generic bucket labels such as Definition, Conditions, Assessment, Applications,
Foundations, or Implications. Preserve any valid, distinct, concept-specific child verbatim and
replace only duplicates, generic buckets, paraphrases, or inaccurate material.
Do not reuse the parent title with a generic prefix or suffix. For Double Effect, do not generate
Principle of Double Effect, Intended Good Effect, or Foreseen Harmful Effect as the three
dimensions. Use these accurate distinctions:
- Moral Object: what the chosen act is in itself; it must be good or morally neutral independently
  of the intended good effect.
- Intention and Means-End Order: the harmful effect is not intended, and the good effect is not
  achieved through the harmful effect.
- Proportionate Reason: permitting the foreseen harm requires a sufficiently serious reason
  proportionate to that harm.
{MINIMAL_REPAIR_INSTRUCTION}

Original Node Concept:
{json.dumps(source, ensure_ascii=False, indent=2)}

Previous output:
{invalid_output}

Required shape:
{{"children":[
  {{"title":"...","definition":"...","example":"..."}},
  {{"title":"...","definition":"...","example":"..."}},
  {{"title":"...","definition":"...","example":"..."}}
]}}
</TASK:REPAIR_MAKE_NODE_CHILDREN>
""".strip()

    @staticmethod
    def _tree_analysis_repair_prompt(
        invalid_output: str,
        question: str,
        response: str,
    ) -> str:
        pivotal_instruction = (
            """
The repaired subject label must be "Conscience". The candidate must use a noun-phrase label naming
culpability and moral object, and its summary must state that an erroneous conscience may affect
culpability without changing the act's moral object.
""".strip()
            if (
                "culpability" in response.casefold()
                and "moral object" in response.casefold()
                and "conscience" in response.casefold()
            )
            else ""
        )
        return f"""
<TASK:REPAIR_INSIGHT_TREE_JSON>
Convert the previous output into one valid JSON object with keys subject and insight_candidate.
subject must contain label and summary. insight_candidate must be either null or one object with
label, summary, and evidence_excerpt. The evidence_excerpt must be copied exactly from the answer.
Each non-empty label must contain one to five words and use as few words as possible.
Use null for an ungrounded seed.
Preserve every valid substantive field verbatim. If the previous subject is a valid string instead
of an object, preserve that exact string as subject.label and derive only its missing concise
summary from the Question and Answer. Do not replace a valid subject stylistically. Candidate
labels must be noun or noun phrases, never clauses or mini-claims.
{pivotal_instruction}
{MINIMAL_REPAIR_INSTRUCTION}

Required shape:
{{
  "subject": {{"label": "One to five words", "summary": "One concise sentence."}},
  "insight_candidate": null
}}
When a grounded candidate is retained, replace null with an object containing label, summary, and
an exact evidence_excerpt from Answer.

Question:
{question}

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
        required_insight_term: str | None = None,
    ) -> str:
        thinking_instruction = (
            """
thinking_summary must contain 1-3 short, inquiry-specific, neutral approach notes, using only as
many as the inquiry warrants. Name the real concepts, comparisons, evidence, or uncertainty being
considered in concise active language. Do not reveal private scratch work or hidden
chain-of-thought, use generic filler, repeat the answer, reveal its conclusion prematurely, or
claim to inspect sources, data, or tools that were not actually used.
""".strip()
            if thinking_enabled
            else (
                "The user-facing approach summary is disabled, so thinking_summary must be an "
                "empty array."
            )
        )
        insight_instruction = (
            f"""
The latest user request directly asks for a definition of "{required_insight_term}". insight must
be a complete object for that term with only title and a concise, self-contained definition in the
immediate conversation or passage's specific sense; it must not be null. Prefer one precise
distinction over loose synonyms, and do not invent an author, quotation, source, earlier statement,
or surrounding context. Do not include pronunciation, part of speech, or an example. The client
renders the Insight card before response, so response must remain a substantive continuation that
answers the inquiry beyond the card rather than a lead-in or repetition of the definition.
Use exactly this shape:
{{"title": "{required_insight_term}", "definition": "A concise contextual definition."}}
""".strip()
            if required_insight_term is not None
            else """
insight must be null unless the user explicitly requested an Insight; when present it must contain
only title and a concise, self-contained contextual definition. Do not include pronunciation,
part of speech, or an example. When insight is present, response must still substantively answer
the inquiry after the client-rendered card rather than merely introducing or repeating it.
""".strip()
        )
        return f"""
<TASK:REPAIR_CONVERSATION_JSON>
Convert the previous response into exactly one valid JSON object with no Markdown or commentary.
Required keys are response, thinking_summary, key_terms, and insight.
{MINIMAL_REPAIR_INSTRUCTION}
{thinking_instruction}
key_terms must be an array of objects with display_text,
canonical_term, and context_excerpt. Highlighted text and excerpts must be copied exactly from
response. Remove an invalid key term rather than altering valid response prose to accommodate it.
{insight_instruction} When a required Insight is missing, derive it only from the named term and
the substantive content already present in the previous response. Preserve valid H1, H2,
ordered-list, and unordered-list Markdown inside response.

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
{MINIMAL_REPAIR_INSTRUCTION}
Preserve the intended contextual meaning while removing any example, pronunciation, or
part-of-speech fields.

Selected term: {term}
Previous response:
{invalid_output}

Required keys:
title, context, definition
</TASK:REPAIR_JSON>
""".strip()

    @classmethod
    def _parse_definition(
        cls,
        output: str,
        fallback_title: str,
        require_context: bool = False,
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
        context = cls._clean_string(
            data.get("context") or data.get("context_label"),
            required=False,
        )

        if not title:
            raise StructuredGenerationError("The model omitted the definition title.")
        if not definition:
            raise StructuredGenerationError("The model omitted the contextual definition.")
        if require_context and not context:
            raise StructuredGenerationError("The model omitted the definition context.")

        return ContextualDefinition(
            title=title,
            part_of_speech=part_of_speech,
            pronunciation=pronunciation,
            definition=definition,
            example=example,
            context=context,
        )

    @classmethod
    def _parse_concept_blend_candidates(
        cls,
        output: str,
    ) -> tuple[ContextualDefinition, ...]:
        data = cls._extract_json_object(output)
        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) != 5:
            raise StructuredGenerationError(
                "Midpoint generation must return exactly five candidates."
            )

        candidates: list[ContextualDefinition] = []
        seen_titles: set[str] = set()
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise StructuredGenerationError(
                    "Every Midpoint candidate must be an object."
                )
            candidate = cls._parse_definition(
                json.dumps(raw_candidate, ensure_ascii=False),
                fallback_title="Blended Insight",
            )
            normalized_title = candidate.title.casefold()
            if normalized_title in seen_titles:
                raise StructuredGenerationError(
                    "Midpoint candidate titles must be distinct."
                )
            seen_titles.add(normalized_title)
            candidates.append(
                ContextualDefinition(
                    title=candidate.title,
                    part_of_speech="",
                    pronunciation="",
                    definition=candidate.definition,
                    example=candidate.example,
                    context=candidate.context,
                )
            )
        return tuple(candidates)

    @classmethod
    def _parse_midpoint_center(cls, output: str) -> str:
        data = cls._extract_json_object(output)
        center = cls._clean_string(data.get("center"), required=False)
        if not center:
            raise StructuredGenerationError(
                "The model omitted the integrated Midpoint center."
            )
        return center

    @staticmethod
    def _anchor_midpoint_candidates(
        candidates: Sequence[ContextualDefinition],
        weighted_center: str,
    ) -> tuple[ContextualDefinition, ...]:
        if not weighted_center:
            return tuple(candidates)
        normalized_center = " ".join(weighted_center.casefold().split())
        anchored: list[ContextualDefinition] = []
        for candidate in candidates:
            normalized_definition = " ".join(
                candidate.definition.casefold().split()
            )
            definition = (
                candidate.definition
                if normalized_center in normalized_definition
                else f"{weighted_center} {candidate.definition}"
            )
            anchored.append(
                ContextualDefinition(
                    title=candidate.title,
                    part_of_speech="",
                    pronunciation="",
                    definition=definition,
                    example=candidate.example,
                    context=candidate.context,
                )
            )
        return tuple(anchored)

    @staticmethod
    def _validate_midpoint_relations(
        candidates: Sequence[ContextualDefinition],
    ) -> None:
        forbidden = (
            "compassion as the end of justice",
            "end of justice is compassion",
            "essence of justice is",
            "final purpose of justice",
            "justice must yield to mercy",
            "justice yields to mercy",
        )
        for candidate in candidates:
            text = f"{candidate.title} {candidate.definition}".casefold()
            if any(phrase in text for phrase in forbidden):
                raise StructuredGenerationError(
                    "A Midpoint candidate improperly collapses one source "
                    "into another."
                )

    @staticmethod
    def _midpoint_source_represented(
        source: ContextualDefinition,
        text: str,
    ) -> bool:
        normalized_text = " ".join(text.casefold().split())
        normalized_title = " ".join(source.title.casefold().split())
        if normalized_title and normalized_title in normalized_text:
            return True
        stopwords = {
            "about",
            "according",
            "concrete",
            "each",
            "from",
            "into",
            "particular",
            "that",
            "their",
            "things",
            "through",
            "what",
            "which",
            "with",
        }

        def roots(value: str) -> set[str]:
            words = re.findall(r"[a-z]+", value.casefold())
            rooted: set[str] = set()
            for word in words:
                if len(word) <= 3 or word in stopwords:
                    continue
                for suffix in ("ing", "ed", "es", "s"):
                    if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                        word = word[: -len(suffix)]
                        break
                rooted.add(word)
            return rooted

        definition_roots = roots(source.definition)
        text_roots = roots(normalized_text)
        return len(definition_roots & text_roots) >= min(
            2,
            len(definition_roots),
        )

    @staticmethod
    def _validate_midpoint_source_coverage(
        candidates: Sequence[ContextualDefinition],
        weighted_sources: Sequence[tuple[ContextualDefinition, float]],
    ) -> None:
        ordered_sources = sorted(
            weighted_sources,
            key=lambda item: item[1],
            reverse=True,
        )
        active_sources = tuple(
            source
            for source, _weight in ordered_sources
            if " ".join(source.title.split())
        )
        dominant_sources = active_sources[: min(2, len(active_sources))]
        for candidate in candidates:
            candidate_text = " ".join(
                f"{candidate.title} {candidate.definition}".casefold().split()
            )
            missing = [
                source.title
                for source in dominant_sources
                if not AquinasGenerationService._midpoint_source_represented(
                    source,
                    candidate_text,
                )
            ]
            if missing:
                raise StructuredGenerationError(
                    "Every Midpoint candidate must explicitly integrate the "
                    f"dominant sources; missing: {', '.join(missing)}."
                )
        if len(active_sources) < 5:
            return
        pool_text = " ".join(
            f"{candidate.title} {candidate.definition}".casefold()
            for candidate in candidates
        )
        missing_from_pool = [
            source.title
            for source in active_sources
            if not AquinasGenerationService._midpoint_source_represented(
                source,
                pool_text,
            )
        ]
        if missing_from_pool:
            raise StructuredGenerationError(
                "The Midpoint candidate pool must cover every nonzero source; "
                f"missing: {', '.join(missing_from_pool)}."
            )

    @classmethod
    def _parse_concept_children(
        cls,
        output: str,
        parent: ContextualDefinition | None = None,
    ) -> tuple[ContextualDefinition, ...]:
        data = cls._extract_json_object(output)
        raw_children = data.get("children")
        if not isinstance(raw_children, list) or len(raw_children) != 3:
            raise StructuredGenerationError(
                "Make Node must return exactly three children."
            )

        children: list[ContextualDefinition] = []
        seen_titles: set[str] = set()
        for raw_child in raw_children:
            if not isinstance(raw_child, dict):
                raise StructuredGenerationError(
                    "Every Make Node child must be an object."
                )
            title = cls._clean_string(raw_child.get("title"))
            definition = cls._clean_string(
                raw_child.get("definition") or raw_child.get("meaning")
            )
            example = cls._clean_string(
                raw_child.get("example"),
                required=False,
            )
            normalized_title = title.casefold()
            normalized_definition = " ".join(definition.casefold().split()).strip(
                " ."
            )
            if not title or not definition:
                raise StructuredGenerationError(
                    "Every Make Node child needs a title and definition."
                )
            if normalized_title in seen_titles:
                raise StructuredGenerationError(
                    "Make Node child titles must be distinct."
                )
            if parent is not None:
                normalized_parent = " ".join(
                    parent.definition.casefold().split()
                ).strip(" .")
                generic_titles = {
                    "applications",
                    "assessment",
                    "conditions",
                    "definition",
                    "foundations",
                    "implications",
                }
                if normalized_title in generic_titles:
                    raise StructuredGenerationError(
                        "Make Node child titles must be concept-specific."
                    )
                if normalized_definition == normalized_parent:
                    raise StructuredGenerationError(
                        "A Make Node child must not restate the parent definition."
                    )
            seen_titles.add(normalized_title)
            children.append(
                ContextualDefinition(
                    title=title,
                    part_of_speech="",
                    pronunciation="",
                    definition=definition,
                    example=example,
                )
            )
        return tuple(children)

    @classmethod
    def _parse_conversation_response(
        cls,
        output: str,
    ) -> StructuredConversationResponse:
        data = cls._extract_json_object(output)
        response = cls._clean_markdown_string(
            data.get("response") or data.get("answer")
        )
        if not response:
            raise StructuredGenerationError("The model omitted its response.")

        raw_summary = data.get("thinking_summary", [])
        if not isinstance(raw_summary, list):
            raise StructuredGenerationError("thinking_summary must be an array.")
        thinking_summary = tuple(
            summary
            for item in raw_summary[:3]
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

        raw_insight = data.get("insight")
        insight = None
        if raw_insight is not None:
            if not isinstance(raw_insight, dict):
                raise StructuredGenerationError(
                    "insight must be an object or null."
                )
            title = cls._clean_string(
                raw_insight.get("title") or raw_insight.get("word")
            )
            definition = cls._clean_string(
                raw_insight.get("definition") or raw_insight.get("meaning")
            )
            if not title or not definition:
                raise StructuredGenerationError(
                    "A generated Insight needs a title and definition."
                )
            insight = ContextualDefinition(
                title=title,
                part_of_speech=cls._clean_string(
                    raw_insight.get("part_of_speech"),
                    required=False,
                ),
                pronunciation=cls._clean_string(
                    raw_insight.get("pronunciation"),
                    required=False,
                ),
                definition=definition,
                example=cls._clean_string(
                    raw_insight.get("example"),
                    required=False,
                ),
            )

        return StructuredConversationResponse(
            response=response,
            thinking_summary=thinking_summary,
            key_terms=tuple(terms),
            insight=insight,
        )

    @classmethod
    def _parse_daily_question(
        cls,
        output: str,
        allowed_insight_titles: Sequence[str],
    ) -> GeneratedDailyQuestion:
        data = cls._extract_json_object(output)
        question = cls._clean_string(data.get("question"))
        rationale = cls._clean_string(data.get("reason_for_asking"))
        if not question or not rationale:
            raise StructuredGenerationError(
                "A daily question requires a question and rationale."
            )
        if not question.endswith("?"):
            raise StructuredGenerationError("The daily question must end with a question mark.")

        requested_title = cls._clean_string(
            data.get("cited_insight_title"),
            required=False,
        )
        title_lookup = {
            title.casefold(): title
            for title in allowed_insight_titles
            if cls._clean_string(title, required=False)
        }
        cited_title = title_lookup.get(requested_title.casefold()) if requested_title else None
        return GeneratedDailyQuestion(
            question=question,
            rationale=rationale,
            cited_insight_title=cited_title,
        )

    @classmethod
    def _parse_quote_notability(cls, output: str) -> GeneratedQuoteNotability:
        data = cls._extract_json_object(output)
        is_notable_insight = data.get("is_notable_insight")
        if not isinstance(is_notable_insight, bool):
            raise StructuredGenerationError(
                "is_notable_insight must be a JSON boolean."
            )
        reason = cls._clean_string(data.get("reason"), required=False)
        return GeneratedQuoteNotability(
            is_notable_insight=is_notable_insight,
            reason=reason or None if is_notable_insight else None,
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
        if label and len(label.split()) > 5:
            raise StructuredGenerationError(
                "The subject label must contain one to five words."
            )

        raw_candidate = data.get("insight_candidate")
        if raw_candidate is not None and not isinstance(raw_candidate, dict):
            raise StructuredGenerationError(
                "insight_candidate must be an object or null."
            )

        insight_candidate = None
        if isinstance(raw_candidate, dict):
            candidate_label = cls._canonical_tree_title(
                cls._clean_string(raw_candidate.get("label"), required=False)
            )
            candidate_summary = cls._clean_string(
                raw_candidate.get("summary"),
                required=False,
            )
            requested_evidence = cls._clean_string(
                raw_candidate.get("evidence_excerpt"),
                required=False,
            )
            evidence = cls._exact_substring(response, requested_evidence)
            padded_label = f" {candidate_label.casefold()} "
            sentence_shaped_label = any(
                marker in padded_label
                for marker in (
                    " are ",
                    " can ",
                    " cannot ",
                    " do ",
                    " does ",
                    " is ",
                    " must ",
                    " should ",
                    " will ",
                )
            )
            requires_culpability_distinction = (
                "culpability" in response.casefold()
                and "moral object" in response.casefold()
                and "conscience" in response.casefold()
            )
            preserves_culpability_distinction = (
                "culpability" in candidate_label.casefold()
                and "object" in candidate_label.casefold()
                and "culpability" in candidate_summary.casefold()
                and "moral object" in candidate_summary.casefold()
            )
            if sentence_shaped_label or (
                requires_culpability_distinction
                and not preserves_culpability_distinction
            ):
                raise StructuredGenerationError(
                    "The Tree candidate must use a nominal label and preserve "
                    "the pivotal distinction."
                )
            if (
                candidate_label
                and candidate_summary
                and evidence
                and not candidate_label.endswith("?")
                and not candidate_summary.endswith("?")
                and not cls._is_non_durable_tree_candidate(
                    response=response,
                    evidence=evidence,
                )
                and len(candidate_label.split()) <= 5
            ):
                insight_candidate = GeneratedTreeInsightCandidate(
                    label=candidate_label,
                    summary=candidate_summary,
                    evidence_excerpt=evidence,
                )

        if insight_candidate and (not label or not summary):
            raise StructuredGenerationError(
                "A tree update with an Insight candidate requires a subject label and summary."
            )
        return GeneratedTreeUpdate(
            subject_label=label,
            subject_summary=summary,
            insight_candidate=insight_candidate,
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
        if (
            not label
            or not any(character.isalnum() for character in label)
            or len(label.split()) > 5
        ):
            raise StructuredGenerationError("The model returned an invalid Node subject.")
        return GeneratedNodeSubject(label=label)

    @classmethod
    def _parse_distinct_node_subject(
        cls,
        output: str,
        insight_descriptions: Sequence[str],
    ) -> GeneratedNodeSubject:
        subject = cls._parse_node_subject(output)

        def canonical(value: str) -> str:
            return " ".join(
                cls._canonical_tree_title(value).casefold().split()
            ).strip(" .,:;!?")

        label_key = canonical(subject.label)
        insight_title_keys = {
            canonical(description.partition(":")[0])
            for description in insight_descriptions
        }
        if label_key in insight_title_keys:
            raise StructuredGenerationError(
                "A Node Concept label must differ from its Insight titles."
            )
        return subject

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

    @classmethod
    def _is_trivial_tree_turn(cls, question: str, response: str) -> bool:
        normalized_question = " ".join(question.casefold().split())
        normalized_response = " ".join(response.casefold().split())
        if cls._is_non_durable_tree_candidate(response, response):
            return True
        procedural_question_markers = (
            "what to click",
            "what do i click",
            "what should i click",
            "what to tap",
            "what do i tap",
            "what should i tap",
            "what to press",
            "what do i press",
            "what should i press",
            "what now",
        )
        procedural_response_prefixes = (
            "click ",
            "open ",
            "press ",
            "select ",
            "tap ",
            "type ",
        )
        return (
            any(marker in normalized_question for marker in procedural_question_markers)
            and normalized_response.startswith(procedural_response_prefixes)
        )

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
    def _clean_markdown_string(value) -> str:
        if not isinstance(value, str):
            raise StructuredGenerationError("The response must be a string.")

        # Normalize whitespace within each Markdown line without flattening the
        # line boundaries that terminate headings and separate list items.
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_lines = (" ".join(line.split()) for line in normalized.split("\n"))
        return "\n".join(cleaned_lines).strip()

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
