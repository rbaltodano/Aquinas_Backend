import re
import chromadb
from mlx_lm import load, generate

print("--- ⏳ 1. Loading The Librarian (Vector DB)... ---")
# This creates a folder called 'aquinas_memory' on your Mac to permanently save chats
chroma_client = chromadb.PersistentClient(path="./aquinas_memory")
# We create a 'collection' (like a specific notebook) for the logs
memory_bank = chroma_client.get_or_create_collection(name="chat_logs")

print(f"--- ⏳ 2. Loading Aquinas (9GB)... ---")
model_path = "models/Aquinas-Final"
model, tokenizer = load(model_path)
print("--- ✅ Engine Ready. Type 'exit' to quit. ---")

system_instruction = (
    "You are Aquinas, a Thomistic Logic Engine. "
    "CORE PERSONA: You are a warm, seasoned mentor. Your tone is hospitable, patient, and direct. "
    "Speak as if we are sharing a quiet conversation. "
    
    "GUIDELINES:\n"
    "1. For simple talk: Be brief and natural. No formal structures needed. "
    "2. For deep inquiries: Acknowledge the weight of the question with a brief, thoughtful reflection (2-3 sentences) that shows you are listening, then transition into the Scholastic Dialectic. "
    "3. When using the Dialectic (Objections, I answer that): Use it to illuminate the truth, but keep your 'Replies' grounded and clear. "
    
    "Maintain Level 7 brevity. Be profound, but get to the heart of the matter quickly. "
    "Do not think out loud.\n\n"
)

# We use a counter to give every memory a unique ID
turn_counter = memory_bank.count()

while True:
    user_input = input("\nInquiry: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Shutting down the engine.")
        break

    # ==========================================
    # THE RAG SEARCH (Retrieving the past)
    # ==========================================
    past_context = ""
    # Only search if we actually have memories saved!
    if turn_counter > 0:
        # Ask ChromaDB to find the 2 most relevant past exchanges to the new question
        search_results = memory_bank.query(
            query_texts=[user_input],
            n_results=min(2, turn_counter) # Grab up to 2, or less if it's a new chat
        )
        
        # Stitch those retrieved memories into a single string
        if search_results['documents'][0]:
            past_context = "\n[RELEVANT PAST MEMORIES]:\n" + "\n".join(search_results['documents'][0]) + "\n"

    # ==========================================
    # AUGMENTED GENERATION (Building the prompt)
    # ==========================================
    reminder = "\n[SYSTEM REMINDER: You are Aquinas. Evaluate the inquiry using Rule 1 (direct answer) or Rule 2 (strict dialectic).]"
    
    # We package: Instructions + Relevant Past Memories + New Question + Reminder
    messages = [{"role": "user", "content": system_instruction + past_context + "\nNew Inquiry: " + user_input + reminder}]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("Thinking...\n")
    response = generate(model, tokenizer, prompt=prompt, max_tokens=1200, verbose=False)
    clean_response = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
    
    print(clean_response)

    # ==========================================
    # SAVING THE MEMORY (Archiving to disk)
    # ==========================================
    # We combine the exact question and answer into one solid memory block
    full_exchange = f"User asked: '{user_input}'\nAquinas answered: '{clean_response}'"
    
    # Save it to the ChromaDB folder on your hard drive
    turn_counter += 1
    memory_bank.add(
        documents=[full_exchange],
        ids=[str(turn_counter)]
    )
