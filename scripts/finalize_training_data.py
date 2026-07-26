import json
import re
import os

# 1. Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
SOURCE_NAME = "summa_theologica"
OUTPUT_FILE = "aquinas_train.jsonl" # Note the .jsonl extension

def process_to_jsonl():
    source_path = os.path.join(RAW_DATA_DIR, SOURCE_NAME)
    if not os.path.exists(source_path):
        source_path += ".txt"
        if not os.path.exists(source_path):
            print("Source file not found.")
            return

    with open(source_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Pre-cleaning
    clean_text = re.sub(r'^\s*\d+\s*$', '', raw_text, flags=re.MULTILINE)

    # Split by Article
    article_pattern = r"(Article\.?\s+\d+\s*-\s*.*?)(?=Article\.?\s+\d+\s*-|$)"
    matches = re.findall(article_pattern, clean_text, re.DOTALL)

    print(f"Processing {len(matches)} articles into JSONL...")

    # 4. Save as JSON Lines
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for match in matches:
            # Clean up the text for this specific line
            full_text = " ".join([line.strip() for line in match.split('\n') if line.strip()])
            
            # Create a single dictionary for this line
            line_data = {"text": full_text}
            
            # Write the dictionary as a single line of JSON
            f.write(json.dumps(line_data, ensure_ascii=False) + "\n")

    print(f"Success! '{OUTPUT_FILE}' is now in the format you wanted.")

if __name__ == "__main__":
    process_to_jsonl()
