import re
import time
from threading import Lock

from collections.abc import Iterator

from mlx_lm import generate, load, stream_generate

model_path = "models/Aquinas-Final"

print(f"--- ⏳ 1. Loading 9GB Model... ---")
start_time = time.time()
model, tokenizer = load(model_path)
print(f"--- ✅ Loaded in {time.time() - start_time:.1f}s ---")

SYSTEM_INSTRUCTION = (
    "You are Aquinas, a Thomistic Logic Engine. "
    "CORE PERSONA: You are a warm, seasoned mentor. Your tone is hospitable, patient, and direct. "
    "Speak as if we are sharing a quiet conversation. "
    "GUIDELINES:\n"
    "1. For simple talk: Be brief and natural. No formal structures needed. "
    "2. For deep inquiries: Acknowledge the weight of the question with a brief, thoughtful "
    "reflection (2-3 sentences) that shows you are listening, then transition into the "
    "Scholastic Dialectic. "
    "3. When using the Dialectic (Objections, I answer that): Use it to illuminate the truth, "
    "but keep your 'Replies' grounded and clear. "
    "Maintain Level 7 brevity. Be profound, but get to the heart of the matter quickly. "
    "Do not think out loud.\n\n"
)

_generation_lock = Lock()


def generate_aquinas(instruction: str, max_tokens: int = 1_200) -> str:
    """Run one serialized model generation for chat or a structured task."""
    messages = [
        {
            "role": "user",
            "content": SYSTEM_INSTRUCTION + instruction,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"--- 🧠 2. Deep Thinking (Letting the engine do its math)... ---")

    with _generation_lock:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )

    # Remove a hidden thought channel if this model emits one.
    clean_response = re.sub(
        r"<\|channel>thought.*?<channel\|>",
        "",
        response,
        flags=re.DOTALL,
    )
    return clean_response.strip()


def generate_aquinas_stream(
    instruction: str,
    max_tokens: int = 1_200,
) -> Iterator[str]:
    """Yield decoded model text as MLX produces it.

    Callers are responsible for exposing only application-approved fields from
    structured output. In particular, this generator must not be sent directly
    to a client because a checkpoint may emit private scratch-work channels.
    """
    messages = [
        {
            "role": "user",
            "content": SYSTEM_INSTRUCTION + instruction,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("--- 🧠 2. Streaming generation... ---")

    with _generation_lock:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            if response.text:
                yield response.text


def ask_aquinas(query: str) -> str:
    return generate_aquinas(f"Inquiry: {query}")

if __name__ == "__main__":
    print("--- ⚔️ 3. Engine Ready. ---")
    user_input = "Whether it is virtuous for a web designer to use CSS variables?"
    print(f"\nINQUIRY: {user_input}\n" + "="*20)
    print(ask_aquinas(user_input))
