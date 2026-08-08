import queue
import re
import threading
import time

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Sequence

from mlx_vlm import apply_chat_template, generate, load, stream_generate
from PIL import Image

from generation_coordinator import (
    GenerationCoordinator,
    GenerationPreempted,
)
from model_identity import (
    MODEL_ADAPTER_PATH,
    MODEL_DISPLAY_NAME,
    MODEL_RUNTIME_PATH,
)

# MLX tracks its execution stream per-thread, and internal state created
# during model load (KV-cache buffers, compiled graphs) stays bound to the
# thread that created it — reusing it from a different thread later fails
# with "There is no Stream(cpu, 0) in current thread" even after that other
# thread has its own stream registered. Loading the model and running every
# later generation call on this exact same dedicated thread avoids the
# cross-thread handoff entirely, rather than trying to make hand-off safe.
#
# Stream setup itself is done lazily inside _ensure_mlx_stream_ready, called
# only from the real generation call sites below — not as an executor-wide
# initializer. Tests load this module with mlx_vlm replaced by a fake object
# (see test_prompt_assembly.py) purely to exercise import-time guards, and
# must never be forced to touch the real mlx.core just because a fake load()
# happened to run on this thread.
_mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-generation")
_mlx_thread_local = threading.local()


def _run_on_mlx_thread(fn, *args, **kwargs):
    return _mlx_thread.submit(fn, *args, **kwargs).result()


def _ensure_mlx_stream_ready() -> None:
    if getattr(_mlx_thread_local, "stream_ready", False):
        return
    import mlx.core as mx

    mx.set_default_stream(mx.new_stream(mx.default_device()))
    _mlx_thread_local.stream_ready = True


print(
    f"--- ⏳ 1. Loading Aquinas ({MODEL_DISPLAY_NAME}) from "
    f"{MODEL_RUNTIME_PATH} with {MODEL_ADAPTER_PATH}... ---"
)
start_time = time.time()
model, processor = _run_on_mlx_thread(
    load,
    MODEL_RUNTIME_PATH,
    adapter_path=MODEL_ADAPTER_PATH,
)
tokenizer = getattr(processor, "tokenizer", processor)
print(f"--- ✅ {MODEL_DISPLAY_NAME} loaded in {time.time() - start_time:.1f}s ---")

LEGACY_TOKENIZER_IDENTITY_MARKERS = (
    "Thomistic Logic Engine",
    "DO NOT PLAN",
    "Assistant: Objection 1:",
)


def _assert_no_legacy_tokenizer_identity(loaded_tokenizer) -> None:
    init_kwargs = getattr(loaded_tokenizer, "init_kwargs", {}) or {}
    configured_system_prompt = init_kwargs.get("system_prompt", "")
    configured_chat_template = getattr(loaded_tokenizer, "chat_template", "") or ""
    hidden_configuration = f"{configured_system_prompt}\n{configured_chat_template}"
    if any(
        marker in hidden_configuration
        for marker in LEGACY_TOKENIZER_IDENTITY_MARKERS
    ):
        raise RuntimeError(
            "The tokenizer contains a legacy identity override. "
            "Aquinas identity must be supplied by the runtime system instruction."
        )


_assert_no_legacy_tokenizer_identity(tokenizer)

REASONING_CONSTITUTION = (
    "Follow reasoning wherever it leads. Treat your previous claims as revisable positions, not "
    "commitments to defend. Evaluate the strongest reasonable version of the user's argument and "
    "judge both your reasoning and the user's by the same intellectual standard. Distinguish a "
    "contradiction or invalid inference from a disputed premise, missing empirical evidence, or "
    "difference in definitions. When the user supplies reasoning that defeats a premise, exposes "
    "a contradiction, introduces decisive evidence, or supports a better distinction, explicitly "
    "revise the affected conclusion and explain what changed. State the earlier claim that failed, "
    "replace it with the narrowest corrected claim the argument supports, and identify the decisive "
    "reason. Begin with a direct acknowledgment when the user's objection succeeds; do not call a "
    "real contradiction merely apparent or present the correction as a defense of the old wording. "
    "Keep the correction concise. Revise only as far as the argument warrants, while carrying the "
    "revision through any conclusions that depend on it. Do not introduce auxiliary theories, "
    "causal claims, or technical machinery that the correction does not require. Preserve category "
    "distinctions instead of saving a conclusion by relabeling an act, faculty, cause, end, habit, "
    "or disposition. A canonical example: the counterexample that an act flowing from a deliberately "
    "acquired habit can be voluntary in its cause without a fresh explicit choice defeats the "
    "universal claim that every voluntary act must be explicitly chosen when it occurs. The proper "
    "revision distinguishes an act voluntary in itself through present choice from one voluntary in "
    "its cause through a relevant prior voluntary act; it must not conclude that the defeated "
    "universal premise still holds by redefining the second act as non-voluntary. For a successful "
    "correction, prefer one to three compact paragraphs. Do not "
    "revise merely because the user disagrees, insists, or sounds confident, and never defend an "
    "earlier answer merely for consistency or authority. Keep confidence proportionate to the "
    "available reasons and evidence. "
)

APPLICATION_TASK_INSTRUCTION = (
    "You are Aquinas. "
    "Never reveal private reasoning or hidden scratch work. "
    "When an action-specific task is provided, follow its instructions and output format exactly; "
    "the task takes priority over conversational structure. Use neutral, clear editorial language "
    "for structured application tasks such as definitions, Insight and Node generation, labeling, "
    "question generation, compaction, extraction, and repair unless the task explicitly requests "
    "the conversational persona. Never invent a citation, quotation, source location, or "
    "attribution in any application task."
)

SYSTEM_INSTRUCTION = (
    REASONING_CONSTITUTION
    + APPLICATION_TASK_INSTRUCTION
)

generation_coordinator = GenerationCoordinator()


def _prompt_for(
    instruction: str,
    response_prefix: str = "",
    image_count: int = 0,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]
    return apply_chat_template(
        processor,
        model.config,
        messages,
        add_generation_prompt=True,
        num_images=image_count,
    ) + response_prefix


def _image_inputs(images: Sequence[bytes]) -> list[Image.Image] | None:
    return [
        Image.open(BytesIO(image)).convert("RGB")
        for image in images
    ] or None


def generate_aquinas(
    instruction: str,
    max_tokens: int = 1_200,
    *,
    images: Sequence[bytes] = (),
) -> str:
    """Run one serialized model generation for chat or a structured task."""
    prompt = _prompt_for(instruction, image_count=len(images))

    print(f"--- 🧠 2. Deep Thinking (Letting the engine do its math)... ---")

    with generation_coordinator.session("foreground") as lease:
        print(
            f'{{"event":"generation_start","priority":"foreground",'
            f'"queue_wait_seconds":{lease.queue_wait_seconds:.3f}}}'
        )
        def _generate() -> str:
            _ensure_mlx_stream_ready()
            return generate(
                model,
                processor,
                prompt=prompt,
                image=_image_inputs(images),
                max_tokens=max_tokens,
                verbose=False
            ).text

        response = _run_on_mlx_thread(_generate)

    # Remove a hidden thought channel if this model emits one.
    clean_response = re.sub(
        r"<\|channel>thought.*?<channel\|>",
        "",
        response,
        flags=re.DOTALL,
    )
    return clean_response.strip()


def generate_aquinas_fast(
    instruction: str,
    max_tokens: int = 1_200,
    *,
    images: Sequence[bytes] = (),
) -> str:
    """Generate structured JSON directly without opening a hidden-thought channel."""
    response_prefix = "{"
    prompt = _prompt_for(
        instruction,
        response_prefix=response_prefix,
        image_count=len(images),
    )
    started_at = time.perf_counter()
    with generation_coordinator.session("foreground") as lease:
        def _generate() -> str:
            _ensure_mlx_stream_ready()
            return generate(
                model,
                processor,
                prompt=prompt,
                image=_image_inputs(images),
                max_tokens=max_tokens,
                verbose=False,
            ).text

        response = _run_on_mlx_thread(_generate)
    print(
        f'{{"event":"generation_complete","mode":"fast",'
        f'"queue_wait_seconds":{lease.queue_wait_seconds:.3f},'
        f'"generation_seconds":{time.perf_counter() - started_at:.3f}}}'
    )
    return response_prefix + response


def generate_aquinas_stream(
    instruction: str,
    max_tokens: int = 1_200,
    *,
    response_prefix: str = "",
    priority: str = "foreground",
    images: Sequence[bytes] = (),
) -> Iterator[str]:
    """Yield decoded model text as MLX produces it.

    Callers are responsible for exposing only application-approved fields from
    structured output. In particular, this generator must not be sent directly
    to a client because a checkpoint may emit private scratch-work channels.
    """
    prompt = _prompt_for(
        instruction,
        response_prefix=response_prefix,
        image_count=len(images),
    )

    print("--- 🧠 2. Streaming generation... ---")

    started_at = time.perf_counter()
    first_fragment_at: float | None = None
    generated_tokens = 0
    with generation_coordinator.session(priority) as lease:
        # stream_generate must be iterated on the dedicated MLX thread (see
        # _run_on_mlx_thread), but this function needs to yield chunks back to
        # whatever thread called generate_aquinas_stream. Bridge the two with
        # a queue: a producer job on the MLX thread pushes each chunk (or a
        # terminal error/sentinel), and this generator drains it.
        chunk_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()

        def _produce() -> None:
            try:
                _ensure_mlx_stream_ready()
                for response in stream_generate(
                    model,
                    processor,
                    prompt=prompt,
                    image=_image_inputs(images),
                    max_tokens=max_tokens,
                ):
                    if lease.cancel_event.is_set():
                        # Stop driving the generator forward immediately, on the
                        # same thread that's actually producing tokens, rather
                        # than letting it keep generating after preemption.
                        chunk_queue.put(("preempted", None))
                        return
                    chunk_queue.put(("chunk", response))
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer thread
                chunk_queue.put(("error", exc))
            finally:
                chunk_queue.put((_SENTINEL, None))

        _mlx_thread.submit(_produce)

        while True:
            kind, payload = chunk_queue.get()
            if kind is _SENTINEL:
                break
            if kind == "error":
                raise payload
            if kind == "preempted":
                raise GenerationPreempted(
                    "Background generation was preempted by foreground work."
                )
            response = payload
            generated_tokens = response.generation_tokens
            if response.text:
                if first_fragment_at is None:
                    first_fragment_at = time.perf_counter()
                yield response.text
    completed_at = time.perf_counter()
    print(
        f'{{"event":"generation_complete","mode":"stream",'
        f'"priority":"{priority}","queue_wait_seconds":'
        f'{lease.queue_wait_seconds:.3f},"ttft_seconds":'
        f'{(first_fragment_at - started_at) if first_fragment_at else 0:.3f},'
        f'"generation_seconds":{completed_at - started_at:.3f},'
        f'"generated_tokens":{generated_tokens}}}'
    )


def generate_aquinas_background(
    instruction: str,
    max_tokens: int = 1_200,
) -> str:
    """Run a preemptible background structured generation."""
    return "".join(
        generate_aquinas_stream(
            instruction,
            max_tokens=max_tokens,
            priority="background",
        )
    )


def generate_aquinas_background_fast(
    instruction: str,
    max_tokens: int = 1_200,
) -> str:
    """Run preemptible background generation on the direct-JSON path."""
    response_prefix = "{"
    return response_prefix + "".join(
        generate_aquinas_stream(
            instruction,
            max_tokens=max_tokens,
            response_prefix=response_prefix,
            priority="background",
        )
    )


def ask_aquinas(query: str) -> str:
    return generate_aquinas(f"Inquiry: {query}")

if __name__ == "__main__":
    print("--- ⚔️ 3. Engine Ready. ---")
    user_input = "Whether it is virtuous for a web designer to use CSS variables?"
    print(f"\nINQUIRY: {user_input}\n" + "="*20)
    print(ask_aquinas(user_input))
