from contextlib import asynccontextmanager
import base64
import binascii
import hashlib
from io import BytesIO
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Sequence

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=os.environ.get("AQUINAS_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aquinas.server")

from insight_tree import (
    DEFAULT_MEMBERSHIP_THRESHOLD,
    TreeInsight,
    TreeNode,
    insight_tree_engine,
)
from main import (
    MODEL_DISPLAY_NAME,
    GenerationPreempted,
    ask_aquinas,
    generate_aquinas,
    generate_aquinas_background,
    generate_aquinas_background_fast,
    generate_aquinas_fast,
    generate_aquinas_stream,
)
from grounding_retrieval import grounding_retriever
from relatedness import relatedness_provider
from structured_generation import (
    AquinasGenerationService,
    ConversationGenerationMode,
    ConversationImage,
    ConversationInsightQuote,
    ConversationMessage,
    ConversationPersonality,
    ConversationStreamParser,
    ContextualDefinition,
    DailyQuestionInsight,
    GeneratedDailyQuestion,
    GeneratedQuoteNotability,
    StructuredGenerationError,
    is_heuristically_notable,
    requested_definition_term,
    resolve_conversation_generation_mode,
)
from today_in_history import entry_for_date
from tree_store import DynamicDefinitionRecord, persistent_tree_service, tree_store


@asynccontextmanager
async def lifespan(_: FastAPI):
    relatedness_provider.load()
    grounding_retriever.load()
    tree_store.initialize()
    tree_store.reconcile_saved_automatic_aliases(relatedness_provider)
    yield


def _latest_user_message_text(
    messages: Sequence[ConversationMessage],
) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.text
    return ""


API_KEY_ENV_VAR = "AQUINAS_API_KEY"


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-Aquinas-Api-Key"),
) -> None:
    """A no-op until AQUINAS_API_KEY is set in the environment, so local
    development is unaffected by default -- this server binds 0.0.0.0, so
    it's reachable by anything on the LAN, not just loopback. Set the env
    var before exposing this beyond a trusted network, and have the client
    send a matching X-Aquinas-Api-Key header."""
    expected = os.environ.get(API_KEY_ENV_VAR)
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


app = FastAPI(
    title="Aquinas Logic API",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)
generation_service = AquinasGenerationService(
    generate_aquinas,
    fast_generator=generate_aquinas_fast,
    background_generator=generate_aquinas_background,
    background_fast_generator=generate_aquinas_background_fast,
)

# No browser/cookie client exists for this API (the iOS app calls it
# directly), so credentials are never needed -- allow_credentials=True
# combined with a wildcard origin is actually an invalid combination per the
# CORS spec (browsers reject it outright), so it was silently doing nothing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_MAX_REQUESTS = 30
_rate_limit_lock = Lock()
_rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)

# Single source of truth for the "Loose Thread" homepage section's display
# name, so a future rename is a one-line change rather than a grep-and-replace.
LOOSE_THREAD_DISPLAY_NAME = "Loose Thread"
TODAY_IN_HISTORY_LINK_THRESHOLD = 0.60


def rate_limit_generation(request: Request) -> None:
    """A basic per-client sliding-window limiter on the generation-heavy
    endpoints, to blunt naive resource-exhaustion abuse. Not a substitute for
    a real gateway-level limiter in a multi-tenant deployment."""
    client_key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_limit_lock:
        hits = _rate_limit_hits[client_key]
        while hits and now - hits[0] > _RATE_LIMIT_WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= _RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests -- please slow down.",
            )
        hits.append(now)

# Define the format we expect from the frontend
class QueryRequest(BaseModel):
    query: str


class ConversationImagePayload(BaseModel):
    name: str = Field(default="Image", max_length=200)
    media_type: str = Field(pattern=r"^image/(?:jpeg|png|webp)$")
    data_base64: str = Field(min_length=1, max_length=5_000_000)

    @field_validator("data_base64")
    @classmethod
    def image_must_be_valid_and_bounded(cls, value: str) -> str:
        try:
            data = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Image data must be valid base64.") from error
        if len(data) > 3_750_000:
            raise ValueError("Each image must be 3.75 MB or smaller.")
        try:
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if width * height > 25_000_000:
                    raise ValueError("Image dimensions are too large.")
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("Attachment is not a supported image.") from error
        return value

    def to_domain(self) -> ConversationImage:
        return ConversationImage(
            name=self.name,
            media_type=self.media_type,
            data=base64.b64decode(self.data_base64, validate=True),
        )


class ConversationInsightQuotePayload(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    definition: str = Field(min_length=1, max_length=4_000)

    def to_domain(self) -> ConversationInsightQuote:
        return ConversationInsightQuote(
            title=self.title,
            definition=self.definition,
        )


class ConversationMessagePayload(BaseModel):
    role: str = Field(min_length=1, max_length=20)
    text: str = Field(min_length=1, max_length=8_000)
    images: list[ConversationImagePayload] = Field(default_factory=list, max_length=8)
    insight_quote: ConversationInsightQuotePayload | None = None

    @field_validator("role")
    @classmethod
    def role_must_be_supported(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"user", "assistant"}:
            raise ValueError("Role must be 'user' or 'assistant'.")
        return cleaned

    def to_domain(self) -> ConversationMessage:
        return ConversationMessage(
            role=self.role,
            text=self.text,
            images=tuple(image.to_domain() for image in self.images),
            insight_quote=(
                self.insight_quote.to_domain()
                if self.insight_quote is not None
                else None
            ),
        )


class ContextualDefinitionRequest(BaseModel):
    term: str = Field(min_length=1, max_length=200)
    source_excerpt: str = Field(default="", max_length=4_000)
    recent_messages: list[ConversationMessagePayload] = Field(
        default_factory=list,
        max_length=20,
    )

    @field_validator("term")
    @classmethod
    def term_must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Term cannot be blank.")
        return cleaned


class ContextualDefinitionResponse(BaseModel):
    title: str
    part_of_speech: str
    pronunciation: str
    definition: str
    example: str
    context: str = ""


class MidpointConceptPayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    definition: str = Field(default="", max_length=4_000)
    example: str = Field(default="", max_length=4_000)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Concept title cannot be blank.")
        return cleaned

    def to_domain(self) -> ContextualDefinition:
        return ContextualDefinition(
            title=self.title,
            part_of_speech="",
            pronunciation="",
            definition=self.definition,
            example=self.example,
        )


class MidpointBlendRequest(BaseModel):
    concepts: list[MidpointConceptPayload] = Field(min_length=2, max_length=8)
    weights: list[float] = Field(min_length=2, max_length=8)


class MidpointCandidatesResponse(BaseModel):
    candidates: list[ContextualDefinitionResponse]


class MakeNodeChildrenRequest(BaseModel):
    concept: MidpointConceptPayload


class MakeNodeChildrenResponse(BaseModel):
    children: list[ContextualDefinitionResponse]


class ConversationResponseRequest(BaseModel):
    recent_messages: list[ConversationMessagePayload] = Field(
        min_length=1,
        max_length=20,
    )
    compacted_context: str | None = Field(default=None, max_length=20_000)
    thinking_enabled: bool = False
    generation_mode: ConversationGenerationMode = ConversationGenerationMode.AUTOMATIC
    personality: ConversationPersonality = ConversationPersonality.BALANCED


class ConversationCompactionRequest(BaseModel):
    recent_messages: list[ConversationMessagePayload] = Field(
        default_factory=list,
        max_length=20,
    )
    compacted_context: str | None = Field(default=None, max_length=20_000)


class ConversationCompactionResponse(BaseModel):
    summary: str


class DailyQuestionInsightPayload(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    definition: str = Field(min_length=1, max_length=4_000)


class DailyQuestionRequest(BaseModel):
    conversation_title: str = Field(default="", max_length=300)
    recent_messages: list[ConversationMessagePayload] = Field(
        min_length=1,
        max_length=20,
    )
    insights: list[DailyQuestionInsightPayload] = Field(default_factory=list, max_length=4)


class DailyQuestionResponse(BaseModel):
    question: str
    reason_for_asking: str
    cited_insight_title: str | None = None


class ConversationIdRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)


class LooseThreadResponse(BaseModel):
    node_id: str
    node_label: str
    insight_count: int


class GlossedTermResponse(BaseModel):
    term_key: str
    requested_term: str
    title: str
    part_of_speech: str
    pronunciation: str
    definition: str
    example: str
    context: str = ""


class TodayInHistoryRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    override_date: str | None = Field(
        default=None,
        pattern=r"^\d{2}-\d{2}$",
    )


class TodayInHistoryResponse(BaseModel):
    title: str
    description: str
    related_entity: str
    linked_node_id: str | None = None


class FlagQuoteRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    response_id: str = Field(min_length=1, max_length=128)
    quote_text: str = Field(min_length=1, max_length=4_000)


class YourQuoteResponse(BaseModel):
    response_id: str
    quote_text: str
    source: str
    reason: str | None = None


class GeneratedKeyTermResponse(BaseModel):
    display_text: str
    canonical_term: str
    context_excerpt: str


class StructuredConversationResponsePayload(BaseModel):
    response: str
    thinking_summary: list[str]
    key_terms: list[GeneratedKeyTermResponse]
    insight: ContextualDefinitionResponse | None = None


class SimilarityPair(BaseModel):
    left: str = Field(max_length=4_000)
    right: str = Field(max_length=4_000)

    @field_validator("left", "right")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Text cannot be blank.")
        return cleaned


class SimilarityRequest(BaseModel):
    pairs: list[SimilarityPair] = Field(min_length=1, max_length=128)


class SimilarityResult(BaseModel):
    similarity: float
    relatedness: float
    distance: float


class SimilarityResponse(BaseModel):
    model: str
    dimensions: int
    results: list[SimilarityResult]


class NodeSubjectRequest(BaseModel):
    insight_descriptions: list[str] = Field(min_length=1, max_length=12)


class NodeSubjectResponse(BaseModel):
    label: str


class NodeLabelRequest(BaseModel):
    label: str = Field(min_length=1, max_length=200)


class TreeInsightPayload(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    definition: str = Field(min_length=1, max_length=4_000)

    @field_validator("id", "title", "definition")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Value cannot be blank.")
        return cleaned

    def to_domain(self) -> TreeInsight:
        return TreeInsight(
            id=self.id,
            title=self.title,
            definition=self.definition,
        )


class TreeNodePayload(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=200)
    insights: list[TreeInsightPayload] = Field(min_length=1)

    def to_domain(self) -> TreeNode:
        return TreeNode(
            id=self.id,
            label=self.label,
            insights=tuple(insight.to_domain() for insight in self.insights),
        )


class TreeAssignmentRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    insight: TreeInsightPayload
    existing_nodes: list[TreeNodePayload] = Field(default_factory=list)
    membership_threshold: float = Field(
        default=DEFAULT_MEMBERSHIP_THRESHOLD,
        ge=0.0,
        le=1.0,
    )


class EvaluatedNodeResponse(BaseModel):
    node_id: str
    similarity: float
    distance: float


class TreeAssignmentResponse(BaseModel):
    conversation_id: str
    action: str
    node_id: str
    node_label: str
    relatedness: float
    distance: float
    member_insight_ids: list[str]
    evaluated_nodes: list[EvaluatedNodeResponse]
    needs_generated_label: bool


class PersistentTreeAssignmentRequest(BaseModel):
    insight: TreeInsightPayload
    membership_threshold: float = Field(
        default=DEFAULT_MEMBERSHIP_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    suggested_node_label: str | None = Field(default=None, max_length=200)


class StoredInsightResponse(BaseModel):
    id: str
    title: str
    definition: str
    relatedness: float
    distance: float
    source_type: str = "saved_definition"
    source_response_id: str | None = None
    source_branch_id: str | None = None
    evidence_excerpt: str = ""
    extraction_role: str | None = None


class StoredNodeResponse(BaseModel):
    id: str
    label: str
    summary: str
    needs_generated_label: bool
    origin_node_id: str | None = None
    insights: list[StoredInsightResponse]


class StoredNodeEdgeResponse(BaseModel):
    id: str
    from_node_id: str
    to_node_id: str
    relatedness: float
    distance: float
    is_strong_extra: bool


class TreeSnapshotResponse(BaseModel):
    conversation_id: str
    embedding_model: str
    embedding_version: int
    nodes: list[StoredNodeResponse]
    edges: list[StoredNodeEdgeResponse] = Field(default_factory=list)


class ResponseTreeAnalysisRequest(BaseModel):
    branch_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=8_000)
    response: str = Field(min_length=1, max_length=20_000)

    @field_validator("branch_id", "question", "response")
    @classmethod
    def analysis_values_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Response analysis values cannot be blank.")
        return cleaned


class ResponseTreeAnalysisResponse(BaseModel):
    status: str
    analysis_id: str | None = None
    mutation_id: str | None = None
    added_insight_ids: list[str]
    added_node_ids: list[str]
    tree: TreeSnapshotResponse


def assignment_response(
    conversation_id: str,
    decision,
) -> TreeAssignmentResponse:
    return TreeAssignmentResponse(
        conversation_id=conversation_id,
        action=decision.action,
        node_id=decision.node_id,
        node_label=decision.node_label,
        relatedness=decision.relatedness,
        distance=decision.distance,
        member_insight_ids=list(decision.member_insight_ids),
        evaluated_nodes=[
            EvaluatedNodeResponse(
                node_id=item.node_id,
                similarity=item.similarity,
                distance=item.distance,
            )
            for item in decision.evaluated_nodes
        ],
        needs_generated_label=decision.needs_generated_label,
    )


def validated_conversation_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 128:
        raise HTTPException(
            status_code=400,
            detail="Conversation ID must contain 1–128 characters.",
        )
    return cleaned


def _definition_term_key(term: str) -> str:
    return " ".join(term.casefold().split())


def _definition_source_text(request: ContextualDefinitionRequest) -> str:
    assistant_messages = [
        message.text
        for message in request.recent_messages
        if message.role == "assistant"
    ]
    if assistant_messages:
        return assistant_messages[-1]
    return request.source_excerpt


def _definition_source_hash(source_text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in source_text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _definition_response(record: DynamicDefinitionRecord) -> ContextualDefinitionResponse:
    return ContextualDefinitionResponse(
        title=record.title,
        part_of_speech=record.part_of_speech,
        pronunciation=record.pronunciation,
        definition=record.definition,
        example=record.example,
        context=record.context,
    )


def _flag_quote_if_notable(conversation_id: str, response_id: str, question: str) -> None:
    """"Your Own Quote" Tier 2: an additive check riding inside the existing
    tree-update analysis call. It must never fail or delay that call's own
    result -- a broken or malformed notability check is swallowed here, not
    surfaced as the /analyze route's HTTP error."""
    if not is_heuristically_notable(question):
        return
    try:
        notability: GeneratedQuoteNotability = generation_service.assess_quote_notability(
            question
        )
    except StructuredGenerationError:
        logger.warning(
            "Quote notability check failed for %s/%s; skipping.",
            conversation_id,
            response_id,
        )
        return
    if notability.is_notable_insight:
        persistent_tree_service.save_flagged_quote(
            conversation_id=conversation_id,
            response_id=response_id,
            quote_text=question,
            source="heuristic_llm",
            reason=notability.reason,
        )


# Define the Endpoint
@app.post("/ask", dependencies=[Depends(rate_limit_generation)])
def ask_endpoint(request: QueryRequest):
    logger.info("Received /ask request (%d chars)", len(request.query))
    try:
        answer = ask_aquinas(request.query)
        return {"response": answer}
    except Exception:
        logger.exception("The Logic Engine encountered an error.")
        raise HTTPException(status_code=500, detail="The Logic Engine encountered an error.")


@app.post(
    "/concept/define",
    response_model=ContextualDefinitionResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def define_contextual_term(request: ContextualDefinitionRequest):
    try:
        definition = generation_service.define_term(
            term=request.term,
            source_excerpt=request.source_excerpt,
            recent_messages=[
                message.to_domain()
                for message in request.recent_messages
            ],
        )
        return ContextualDefinitionResponse(
            title=definition.title,
            part_of_speech=definition.part_of_speech,
            pronunciation=definition.pronunciation,
            definition=definition.definition,
            example=definition.example,
            context=definition.context,
        )
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return a valid contextual definition.",
        ) from error


@app.post(
    "/conversation/{conversation_id}/concept/lookup",
    response_model=ContextualDefinitionResponse | None,
)
def lookup_contextual_term_for_conversation(
    conversation_id: str,
    request: ContextualDefinitionRequest,
):
    conversation_id = validated_conversation_id(conversation_id)
    cached = tree_store.load_dynamic_definition(
        conversation_id=conversation_id,
        term_key=_definition_term_key(request.term),
        source_hash=_definition_source_hash(_definition_source_text(request)),
    )
    return None if cached is None else _definition_response(cached)


@app.post(
    "/conversation/{conversation_id}/concept/define",
    response_model=ContextualDefinitionResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def define_contextual_term_for_conversation(
    conversation_id: str,
    request: ContextualDefinitionRequest,
):
    conversation_id = validated_conversation_id(conversation_id)
    source_text = _definition_source_text(request)
    term_key = _definition_term_key(request.term)
    source_hash = _definition_source_hash(source_text)

    cached = tree_store.load_dynamic_definition(
        conversation_id=conversation_id,
        term_key=term_key,
        source_hash=source_hash,
    )
    if cached is not None:
        return _definition_response(cached)

    try:
        definition = generation_service.define_term(
            term=request.term,
            source_excerpt=request.source_excerpt,
            recent_messages=[
                message.to_domain()
                for message in request.recent_messages
            ],
        )
        record = DynamicDefinitionRecord(
            title=definition.title,
            part_of_speech=definition.part_of_speech,
            pronunciation=definition.pronunciation,
            definition=definition.definition,
            example=definition.example,
            context=definition.context,
        )
        persistent_tree_service.save_dynamic_definition(
            conversation_id=conversation_id,
            term_key=term_key,
            source_hash=source_hash,
            requested_term=request.term,
            source_excerpt=request.source_excerpt,
            definition=record,
        )
        return _definition_response(record)
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return a valid contextual definition.",
        ) from error


@app.post(
    "/conversation/respond",
    response_model=StructuredConversationResponsePayload,
    dependencies=[Depends(rate_limit_generation)],
)
def respond_to_conversation(request: ConversationResponseRequest):
    try:
        messages = [
            message.to_domain()
            for message in request.recent_messages
        ]
        grounding_passages = grounding_retriever.retrieve(
            _latest_user_message_text(messages),
            relatedness_provider,
        )
        result = generation_service.respond(
            messages,
            compacted_context=request.compacted_context,
            thinking_enabled=request.thinking_enabled,
            generation_mode=request.generation_mode,
            personality=request.personality,
            grounding_passages=grounding_passages,
        )
        return StructuredConversationResponsePayload(
            response=result.response,
            thinking_summary=list(result.thinking_summary),
            key_terms=[
                GeneratedKeyTermResponse(
                    display_text=term.display_text,
                    canonical_term=term.canonical_term,
                    context_excerpt=term.context_excerpt,
                )
                for term in result.key_terms
            ],
            insight=(
                ContextualDefinitionResponse(
                    title=result.insight.title,
                    part_of_speech=result.insight.part_of_speech,
                    pronunciation=result.insight.pronunciation,
                    definition=result.insight.definition,
                    example=result.insight.example,
                    context=result.insight.context,
                )
                if result.insight is not None
                else None
            ),
        )
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return a valid structured response.",
        ) from error


@app.post(
    "/home/question-of-the-day",
    response_model=DailyQuestionResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def generate_question_of_the_day(request: DailyQuestionRequest):
    try:
        result: GeneratedDailyQuestion = generation_service.generate_daily_question(
            conversation_title=request.conversation_title,
            recent_messages=[
                message.to_domain()
                for message in request.recent_messages
            ],
            insights=[
                DailyQuestionInsight(
                    title=insight.title,
                    definition=insight.definition,
                )
                for insight in request.insights
            ],
        )
        return DailyQuestionResponse(
            question=result.question,
            reason_for_asking=result.rationale,
            cited_insight_title=result.cited_insight_title,
        )
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return a valid Question of the Day.",
        ) from error


@app.post(
    "/home/loose-thread",
    response_model=LooseThreadResponse | None,
)
def find_loose_thread(request: ConversationIdRequest):
    conversation_id = validated_conversation_id(request.conversation_id)
    result = tree_store.find_loose_thread(conversation_id)
    return None if result is None else LooseThreadResponse(**result)


@app.post(
    "/home/glossed-terms",
    response_model=GlossedTermResponse | None,
)
def find_glossed_term(request: ConversationIdRequest):
    conversation_id = validated_conversation_id(request.conversation_id)
    result = tree_store.find_glossed_term(conversation_id)
    return None if result is None else GlossedTermResponse(**result)


@app.post(
    "/home/today-in-history",
    response_model=TodayInHistoryResponse | None,
)
def today_in_history(request: TodayInHistoryRequest):
    conversation_id = validated_conversation_id(request.conversation_id)
    month_day = request.override_date or time.strftime("%m-%d")
    entry = entry_for_date(month_day)
    if entry is None:
        return None
    linked_node_id = tree_store.find_related_node(
        conversation_id,
        entry.related_entity,
        TODAY_IN_HISTORY_LINK_THRESHOLD,
        relatedness_provider,
    )
    return TodayInHistoryResponse(
        title=entry.title,
        description=entry.description,
        related_entity=entry.related_entity,
        linked_node_id=linked_node_id,
    )


@app.post("/home/flag-quote")
def flag_quote(request: FlagQuoteRequest):
    conversation_id = validated_conversation_id(request.conversation_id)
    response_id = validated_conversation_id(request.response_id)
    persistent_tree_service.save_flagged_quote(
        conversation_id=conversation_id,
        response_id=response_id,
        quote_text=request.quote_text,
        source="user_flagged",
        reason=None,
    )
    return {"status": "flagged"}


@app.post(
    "/home/your-quote",
    response_model=YourQuoteResponse | None,
)
def find_your_quote(request: ConversationIdRequest):
    conversation_id = validated_conversation_id(request.conversation_id)
    result = persistent_tree_service.find_surfaceable_quote(conversation_id)
    return None if result is None else YourQuoteResponse(**result)


@app.post("/conversation/respond/stream", dependencies=[Depends(rate_limit_generation)])
def stream_conversation_response(request: ConversationResponseRequest):
    messages = [message.to_domain() for message in request.recent_messages]
    images = generation_service.conversation_images(messages)
    required_insight_term = requested_definition_term(messages)
    resolved_mode = resolve_conversation_generation_mode(
        messages,
        request.generation_mode,
    )
    grounding_passages = grounding_retriever.retrieve(
        _latest_user_message_text(messages),
        relatedness_provider,
    )

    def event_stream():
        request_started_at = time.perf_counter()
        first_approved_field_at: float | None = None
        try:
            prompt = generation_service.conversation_prompt(
                messages,
                compacted_context=request.compacted_context,
                thinking_enabled=request.thinking_enabled,
                generation_mode=resolved_mode,
                personality=request.personality,
                grounding_passages=grounding_passages,
            )
            parser = ConversationStreamParser()
            response_prefix = (
                "{"
                if resolved_mode == ConversationGenerationMode.FAST
                else ""
            )
            raw_output: list[str] = [response_prefix] if response_prefix else []
            if response_prefix:
                parser.feed(response_prefix)
            has_started = False

            for fragment in generate_aquinas_stream(
                prompt,
                max_tokens=1_800,
                response_prefix=response_prefix,
                images=images,
            ):
                if not has_started:
                    # `generate_aquinas_stream` serializes access to the model.
                    # Emitting start only after its first fragment means the
                    # client can distinguish waiting for that lock from active
                    # generation.
                    has_started = True
                    yield _stream_event(
                        "start",
                        generation_mode=resolved_mode.value,
                    )
                raw_output.append(fragment)
                for update in parser.feed(fragment):
                    if first_approved_field_at is None:
                        first_approved_field_at = time.perf_counter()
                    if update.kind == "thinking_summary":
                        # The checkpoint can still emit a summary after being told
                        # Thinking is disabled. Never serialize that tuple into the
                        # text-only `delta` field: doing so breaks the iOS event
                        # decoder and turns a healthy backend into a false
                        # connection failure.
                        if request.thinking_enabled:
                            yield _stream_event(
                                update.kind,
                                thinking_summary=list(update.value),
                            )
                        continue
                    yield _stream_event(
                        update.kind,
                        delta=update.value,
                    )

            result = generation_service.parse_conversation_output(
                "".join(raw_output),
                required_insight_term=required_insight_term,
            )
            yield _stream_event(
                "complete",
                response=result.response,
                generation_mode=resolved_mode.value,
                thinking_summary=(
                    list(result.thinking_summary)
                    if request.thinking_enabled
                    else []
                ),
                key_terms=[
                    {
                        "display_text": term.display_text,
                        "canonical_term": term.canonical_term,
                        "context_excerpt": term.context_excerpt,
                    }
                    for term in result.key_terms
                ],
                insight=(
                    {
                        "title": result.insight.title,
                        "part_of_speech": result.insight.part_of_speech,
                        "pronunciation": result.insight.pronunciation,
                        "definition": result.insight.definition,
                        "example": result.insight.example,
                    }
                    if result.insight is not None
                    else None
                ),
            )
            completed_at = time.perf_counter()
            logger.info(
                "conversation_response_complete mode=%s first_approved_field_seconds=%s "
                "total_seconds=%.3f",
                resolved_mode.value,
                (
                    f"{first_approved_field_at - request_started_at:.3f}"
                    if first_approved_field_at is not None
                    else None
                ),
                completed_at - request_started_at,
            )
        except Exception:
            logger.exception("Streaming conversation response failed.")
            yield _stream_event(
                "error",
                detail="Aquinas did not return a valid streaming response.",
            )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/conversation/compact",
    response_model=ConversationCompactionResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def compact_conversation(request: ConversationCompactionRequest):
    try:
        summary = generation_service.compact_context(
            [
                message.to_domain()
                for message in request.recent_messages
            ],
            compacted_context=request.compacted_context,
        )
        return ConversationCompactionResponse(summary=summary)
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return valid compacted context.",
        ) from error


def _stream_event(event_type: str, **payload) -> str:
    return json.dumps({"type": event_type, **payload}) + "\n"


@app.get("/relatedness/health")
def relatedness_health():
    return {
        "status": "ready",
        "model": relatedness_provider.model_name,
        "dimensions": relatedness_provider.dimensions,
    }


@app.get("/health")
def health():
    """Broader liveness check covering everything a request actually
    touches: MLX generation (loaded eagerly at import, so reaching this
    handler at all proves it), the MiniLM relatedness provider, the Chroma
    grounding index, and the SQLite Insight Tree store."""
    checks = {
        "generation_model": {"ok": True, "model": MODEL_DISPLAY_NAME},
        "relatedness_provider": {
            "ok": relatedness_provider.is_loaded,
            "model": relatedness_provider.model_name,
        },
        "grounding_index": {"ok": grounding_retriever.is_loaded},
    }
    try:
        checks["insight_tree_store"] = {"ok": tree_store.ping()}
    except Exception as error:
        logger.exception("Insight Tree store health check failed.")
        checks["insight_tree_store"] = {"ok": False, "detail": str(error)}

    overall_ok = all(check["ok"] for check in checks.values())
    return {"status": "ready" if overall_ok else "degraded", "checks": checks}


@app.post("/relatedness/similarity", response_model=SimilarityResponse)
def relatedness_similarity(request: SimilarityRequest):
    try:
        results = [
            SimilarityResult(**relatedness_provider.compare(pair.left, pair.right))
            for pair in request.pairs
        ]
        return SimilarityResponse(
            model=relatedness_provider.model_name,
            dimensions=relatedness_provider.dimensions,
            results=results,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post(
    "/insight-tree/label-node",
    response_model=NodeSubjectResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def label_insight_tree_node(request: NodeSubjectRequest):
    try:
        subject = generation_service.label_tree_subject(request.insight_descriptions)
        return NodeSubjectResponse(label=subject.label)
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return a valid Node subject.",
        ) from error


@app.post(
    "/concept/blend",
    response_model=MidpointCandidatesResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def blend_midpoint_concepts(request: MidpointBlendRequest):
    if len(request.concepts) != len(request.weights):
        raise HTTPException(
            status_code=400,
            detail="Every Midpoint concept must have one weight.",
        )
    try:
        candidates = generation_service.blend_concept_candidates(
            [concept.to_domain() for concept in request.concepts],
            request.weights,
        )
        return MidpointCandidatesResponse(
            candidates=[
                ContextualDefinitionResponse(
                    title=candidate.title,
                    part_of_speech=candidate.part_of_speech,
                    pronunciation=candidate.pronunciation,
                    definition=candidate.definition,
                    example=candidate.example,
                    context=candidate.context,
                )
                for candidate in candidates
            ]
        )
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return a valid Midpoint Insight.",
        ) from error


@app.post(
    "/concept/children",
    response_model=MakeNodeChildrenResponse,
    dependencies=[Depends(rate_limit_generation)],
)
def generate_make_node_children(request: MakeNodeChildrenRequest):
    try:
        children = generation_service.generate_concept_children(
            request.concept.to_domain()
        )
        return MakeNodeChildrenResponse(
            children=[
                ContextualDefinitionResponse(
                    title=child.title,
                    part_of_speech=child.part_of_speech,
                    pronunciation=child.pronunciation,
                    definition=child.definition,
                    example=child.example,
                    context=child.context,
                )
                for child in children
            ]
        )
    except StructuredGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail="Aquinas did not return three valid Make Node Insights.",
        ) from error


@app.post("/insight-tree/assign", response_model=TreeAssignmentResponse)
def assign_insight_to_node(request: TreeAssignmentRequest):
    try:
        decision = insight_tree_engine.assign_new_insight(
            insight=request.insight.to_domain(),
            nodes=[node.to_domain() for node in request.existing_nodes],
            membership_threshold=request.membership_threshold,
        )
        return assignment_response(request.conversation_id, decision)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post(
    "/insight-tree/{conversation_id}/insights",
    response_model=TreeAssignmentResponse,
)
def save_insight_to_tree(
    conversation_id: str,
    request: PersistentTreeAssignmentRequest,
):
    conversation_id = validated_conversation_id(conversation_id)
    promoted = persistent_tree_service.promote_matching_automatic_insight(
        conversation_id=conversation_id,
        insight=request.insight.to_domain(),
    )
    if promoted is not None:
        persistent_tree_service.mark_definition_promoted(
            conversation_id, _definition_term_key(request.insight.title)
        )
        return assignment_response(conversation_id, promoted)

    existing_tree = tree_store.snapshot(conversation_id)
    for node in existing_tree["nodes"]:
        for stored_insight in node["insights"]:
            if stored_insight["id"] != request.insight.id:
                continue
            if (
                stored_insight["title"] != request.insight.title
                or stored_insight["definition"] != request.insight.definition
            ):
                raise HTTPException(
                    status_code=409,
                    detail="That Insight ID already exists with different content.",
                )
            persistent_tree_service.mark_definition_promoted(
                conversation_id, _definition_term_key(request.insight.title)
            )
            return TreeAssignmentResponse(
                conversation_id=conversation_id,
                action="attached",
                node_id=node["id"],
                node_label=node["label"],
                relatedness=stored_insight["relatedness"],
                distance=stored_insight["distance"],
                member_insight_ids=[
                    insight["id"] for insight in node["insights"]
                ],
                evaluated_nodes=[],
                needs_generated_label=node["needs_generated_label"],
            )
    try:
        decision = persistent_tree_service.assign_and_save(
            conversation_id=conversation_id,
            insight=request.insight.to_domain(),
            membership_threshold=request.membership_threshold,
            suggested_node_label=request.suggested_node_label,
        )
        persistent_tree_service.mark_definition_promoted(
            conversation_id, _definition_term_key(request.insight.title)
        )
        return assignment_response(conversation_id, decision)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get(
    "/insight-tree/{conversation_id}",
    response_model=TreeSnapshotResponse,
)
def read_persistent_tree(conversation_id: str):
    conversation_id = validated_conversation_id(conversation_id)
    return tree_store.snapshot(conversation_id)


@app.post(
    "/insight-tree/{conversation_id}/nodes/{node_id}/label",
    response_model=TreeSnapshotResponse,
)
def set_insight_tree_node_label(
    conversation_id: str,
    node_id: str,
    request: NodeLabelRequest,
):
    conversation_id = validated_conversation_id(conversation_id)
    node_id = validated_conversation_id(node_id)
    try:
        persistent_tree_service.set_node_label(conversation_id, node_id, request.label)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return tree_store.snapshot(conversation_id)


@app.post(
    "/insight-tree/{conversation_id}/responses/{response_id}/analyze",
    dependencies=[Depends(rate_limit_generation)],
    response_model=ResponseTreeAnalysisResponse,
)
def analyze_response_for_tree(
    conversation_id: str,
    response_id: str,
    request: ResponseTreeAnalysisRequest,
):
    conversation_id = validated_conversation_id(conversation_id)
    response_id = validated_conversation_id(response_id)
    cached = tree_store.load_response_analysis(conversation_id, response_id)
    if cached is None:
        try:
            extraction = generation_service.analyze_tree_update(
                question=request.question,
                response=request.response,
                tree_is_empty=not tree_store.has_nodes(conversation_id),
            )
            cached = persistent_tree_service.apply_response_update(
                conversation_id=conversation_id,
                response_id=response_id,
                branch_id=request.branch_id,
                extraction=extraction,
            )
            _flag_quote_if_notable(conversation_id, response_id, request.question)
        except StructuredGenerationError as error:
            raise HTTPException(
                status_code=502,
                detail="Aquinas did not return a valid Insight Tree update.",
            ) from error
        except GenerationPreempted as error:
            raise HTTPException(
                status_code=409,
                detail="Insight Tree analysis was deferred for foreground model work.",
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    return ResponseTreeAnalysisResponse(
        **cached,
        tree=TreeSnapshotResponse(**tree_store.snapshot(conversation_id)),
    )


@app.delete(
    "/insight-tree/{conversation_id}/insights/{insight_id}",
    response_model=TreeSnapshotResponse,
)
def remove_insight_from_tree(conversation_id: str, insight_id: str):
    conversation_id = validated_conversation_id(conversation_id)
    insight_id = validated_conversation_id(insight_id)
    persistent_tree_service.remove_insight(conversation_id, insight_id)
    return tree_store.snapshot(conversation_id)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Aquinas API Server on http://0.0.0.0:8000")
    if not os.environ.get(API_KEY_ENV_VAR):
        logger.warning(
            "%s is not set -- every endpoint on this LAN-reachable server is "
            "unauthenticated. Set %s before exposing this beyond a trusted network.",
            API_KEY_ENV_VAR,
            API_KEY_ENV_VAR,
        )
    uvicorn.run(app, host="0.0.0.0", port=8000)
