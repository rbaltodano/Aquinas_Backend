import json
import os

config_path = "models/Aquinas-Final/tokenizer_config.json"

with open(config_path, "r") as f:
    config = json.load(f)

# THE HAMMER: This prompt explicitly bans "thinking" and "planning"
SYSTEM_PROMPT = (
    "You are the Thomistic Logic Engine. DO NOT PLAN. DO NOT THINK. "
    "Your output MUST begin with the string 'Objection 1:'. "
    "Execute the Summa Theologica dialectic immediately. "
    "Structure: Objection 1, Objection 2, On the contrary, I answer that, Replies. "
    "Use modern terms. Level 7 brevity. No conversational filler."
)

config["system_prompt"] = SYSTEM_PROMPT

# This custom template bypasses the Gemma 'thought' channel by using a
# simple instruction-response format that forces the model to obey.
config["chat_template"] = (
    "{{ bos_token }}"
    "System: " + SYSTEM_PROMPT + "\n\n"
    "User: {{ prompt }}\n"
    "Assistant: Objection 1:"
)

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("--- Identity Force-Locked. Thinking Channel suppressed. ---")
