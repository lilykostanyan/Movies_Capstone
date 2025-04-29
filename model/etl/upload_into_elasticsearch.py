import os, time
import json
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# === Config ===
INDEX_NAME = "movies-bm25-vector"
VECTOR_DIM = 768
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")

# === Connect to Elasticsearch ===
es = Elasticsearch(ES_URL)

# Wait until ES is fully healthy
for i in range(60):
    try:
        if es.ping():
            health = es.cluster.health(wait_for_status="yellow", timeout="30s")
            if health["status"] in ["yellow", "green"]:
                print(f"✅ Elasticsearch is healthy ({health['status']})")
                break
        raise Exception("ES not ready")
    except Exception as e:
        print(f"⏳ Attempt {i+1}/60: waiting for ES... ({e})")
        time.sleep(10)
else:
    raise RuntimeError("❌ ES didn't become healthy in time")

# === Check if Index Exists ===
if es.indices.exists(index=INDEX_NAME):
    print(f"ℹ️ Index '{INDEX_NAME}' already exists. Skipping upload.")
    exit(0)  # Stop the script completely
else:
    # Create the index
    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "movie_id": {"type": "keyword"},
                "genres": {"type": "keyword"},
                "type": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    )
    print(f"📦 Created index: {INDEX_NAME}")

# === Load and Upload JSON Files ===
json_dirs = [
    "./jsons/selected_500_drama_ids_jsons",
    "./jsons/selected_500_rom_ids_jsons"
]

documents_uploaded = 0
files_processed = 0

for json_dir in json_dirs:
    if not os.path.exists(json_dir):
        print(f"⚠️ Warning: Directory '{json_dir}' does not exist. Skipping.")
        continue

    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for filename in files:
        path = os.path.join(json_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)

                # Insert document only if it doesn't exist already
                es.index(index=INDEX_NAME, id=doc["movie_id"], document=doc, op_type="create")

                documents_uploaded += 1
                files_processed += 1
                print(f"✅ Indexed: {filename}")
        except Exception as e:
            # Document already exists or some other error
            print(f"⚠️ Skipped {filename}: {e}")

print(f"🎉 Finished! {documents_uploaded} new documents indexed from {files_processed} files processed.")