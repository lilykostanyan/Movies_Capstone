import os
import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import re
import json

# Load embedding model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), "jsons")
os.makedirs(output_dir, exist_ok=True)

# Load Excel
xlsx_file = "srsly_final_rom_sums.xlsx"

df = pd.read_excel(xlsx_file)

def smart_split_sentences(text):
    abbreviations = {
        "Dr.": "DR_ABBR", "Mr.": "MR_ABBR", "Mrs.": "MRS_ABBR", "Ms.": "MS_ABBR",
        "Jr.": "JR_ABBR", "Sr.": "SR_ABBR", "St.": "ST_ABBR", "Prof.": "PROF_ABBR",
        "Inc.": "INC_ABBR", "Ltd.": "LTD_ABBR", "vs.": "VS_ABBR",
        "e.g.": "EG_ABBR", "i.e.": "IE_ABBR"
    }
    for abbr, token in abbreviations.items():
        text = text.replace(abbr, token)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    restored = []
    for sentence in sentences:
        for abbr, token in abbreviations.items():
            sentence = sentence.replace(token, abbr)
        restored.append(sentence.strip())

    return restored

def chunk_text(text, min_chunk_size=100, max_chunk_size=275, target_chunk_size=240):
    words = text.split()
    if len(words) <= target_chunk_size:
        return [text.strip()]

    sentences = smart_split_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        if chunks and len(" ".join(current_chunk).split()) < min_chunk_size:
            chunks[-1] += " " + " ".join(current_chunk)
        else:
            chunks.append(" ".join(current_chunk).strip())

    return chunks

# Collect all chunks
all_documents = []

for index, row in df.iterrows():
    movie_id = row["tconst"]
    short_synopsis = row["short_synopsis"] if pd.notna(row["short_synopsis"]) else ""
    long_synopsis = row["long_synopsis"] if pd.notna(row["long_synopsis"]) else ""

    # Short synopsis
    if short_synopsis.strip() and short_synopsis.strip() != "No short sum found":
        vector = model.encode(short_synopsis).tolist()
        doc = {
            "movie_id": movie_id,
            "type": "short",
            "chunk_id": f"{movie_id}-sh-1",
            "text": short_synopsis.strip(),
            "vector": vector
        }
        all_documents.append(doc)

    # Summaries
    summaries = row["summaries"]
    if pd.notna(summaries):
        try:
            summaries = eval(summaries) if isinstance(summaries, str) else summaries
        except:
            summaries = [summaries] if isinstance(summaries, str) else []
    else:
        summaries = []

    if summaries and summaries != ['No summaries found']:
        for j, summary in enumerate(summaries):
            if isinstance(summary, str) and summary.strip():
                chunks = chunk_text(summary)
                for i, chunk in enumerate(chunks):
                    vector = model.encode(chunk).tolist()
                    doc = {
                        "movie_id": movie_id,
                        "type": "summary",
                        "chunk_id": f"{movie_id}-summary-{j+1}-{i+1}",
                        "text": chunk,
                        "vector": vector
                    }
                    all_documents.append(doc)

# Long synopsis
    if long_synopsis.strip() and long_synopsis.strip() != "Synopsis not found":
        long_chunks = chunk_text(long_synopsis)
        for i, chunk in enumerate(long_chunks):
            vector = model.encode(chunk).tolist()
            doc = {
                "movie_id": movie_id,
                "type": "long",
                "chunk_id": f"{movie_id}-lon-{i+1}",
                "text": chunk,
                "vector": vector
            }
            all_documents.append(doc)

# Save each chunk as a JSON
for doc in all_documents:
    filename = f"{doc['chunk_id']}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=4)
    print(f"✅ Saved {filename}")

print("🚀 All done!")