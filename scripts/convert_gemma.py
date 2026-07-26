import json
import os

def convert_to_gemma4_format(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"Skipping {filename}: file is empty or missing.")
        return
        
    formatted = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "text" in entry:
                    formatted.append({
                        "messages": [
                            {"role": "user", "content": "Explain this Scholastic concept using the method of the Summa."},
                            {"role": "assistant", "content": entry["text"]}
                        ]
                    })
                else:
                    formatted.append(entry)
            except json.JSONDecodeError:
                print(f"Skipping malformed line {i} in {filename}")

    with open(filename, 'w', encoding='utf-8') as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

for f in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
    convert_to_gemma4_format(f)
    print(f"Verified and formatted {f} for Gemma 4.")
