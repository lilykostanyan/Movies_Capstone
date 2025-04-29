from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery
from google.oauth2 import service_account
import numpy as np
from collections import defaultdict
import json
import re

# === CONFIG ===
ES_INDEX = "romance-vector"
TOP_K = 50         # Top chunks to fetch initially
TOP_MOVIES = 5     # Top movies to recommend
BQ_TABLE = "enduring-brace-451209-q3.romance_dataset.roms_data"
SERVICE_ACCOUNT_FILE = "enduring-brace-451209-q3-35cf0810d57c.json"

# === BIGQUERY AUTH ===
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/bigquery"]
)
bq = bigquery.Client(credentials=credentials, project="enduring-brace-451209-q3")

# === ELASTICSEARCH & MODEL ===
es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer("bert-base-nli-mean-tokens")

# === USER INPUT ===
query = input("🎬 Describe the kind of movie you're looking for: ").strip()
query_vector = model.encode(query).tolist()

# === HYBRID BM25 + VECTOR SEARCH ===
search_body = {
    "size": TOP_K,
    "_source": ["movie_id", "chunk_id", "text", "type"],
    "query": {
        "script_score": {
            "query": {"match": {"text": query}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}
res = es.search(index=ES_INDEX, body=search_body)

# === GROUP SCORES BY MOVIE ===
movie_scores = defaultdict(list)
chunk_meta = defaultdict(list)

for hit in res["hits"]["hits"]:
    doc = hit["_source"]
    movie_id = doc["movie_id"]
    score = hit["_score"]

    movie_scores[movie_id].append(score)
    chunk_meta[movie_id].append({
        "chunk_id": doc["chunk_id"],
        "text": doc["text"],
        "type": doc["type"],
        "score": score
    })

# === RANK MOVIES BY MEAN SCORE ===
avg_scores = {m: np.mean(s) for m, s in movie_scores.items()}
top_movie_ids = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_MOVIES]
top_movie_ids = [mid for mid, _ in top_movie_ids]

# === FETCH TITLES & GENRES FROM BIGQUERY ===
query_str = f"""
SELECT tconst AS movie_id, movie_title, genres
FROM {BQ_TABLE}
WHERE tconst IN UNNEST(@movie_ids)
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter("movie_ids", "STRING", top_movie_ids)]
)
genre_df = bq.query(query_str, job_config=job_config).to_dataframe()
genre_map = dict(zip(genre_df["movie_id"], genre_df["genres"]))
title_map = dict(zip(genre_df["movie_id"], genre_df["movie_title"]))

# === SORT CHUNKS FOR DISPLAY ===
def chunk_sort_key(chunk_id):
    parts = chunk_id.split("-")
    type_order = {"sh": 0, "short": 0, "summary": 1, "lon": 2, "long": 2}
    type_val = type_order.get(parts[1], 99)
    secondary = list(map(int, re.findall(r"\d+", "-".join(parts[2:]))))
    return (type_val, *secondary)

# === DISPLAY RESULTS ===
print("\n🎯 Top Movie Recommendations:\n")

for movie_id in top_movie_ids:
    chunks = chunk_meta[movie_id]
    sorted_chunks = sorted(chunks, key=lambda x: chunk_sort_key(x["chunk_id"]))
    genres = genre_map.get(movie_id, [])

    print(f"🎬 {title_map.get(movie_id, 'Unknown Title')} (ID: {movie_id})")
    print(f"🏷  Genres: {', '.join(genres) if genres else 'Unknown'}")
    print("📖 Preview:")

    for chunk in sorted_chunks[:3]:  # Show top 3 chunks
        print(f"• ({chunk['type']}) {chunk['text'][:200]}...")

    print("-" * 60)


# ####2nd version
# from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer
# from google.cloud import bigquery
# from google.oauth2 import service_account
# import numpy as np
# from collections import defaultdict
# import json
# import re
#
# # === CONFIG ===
# ES_INDEX = "romance-vector"
# TOP_K = 50
# TOP_MOVIES = 5
# SERVICE_ACCOUNT_FILE = "enduring-brace-451209-q3-35cf0810d57c.json"
# BQ_TABLE = "enduring-brace-451209-q3.romance_dataset.roms_data"
# CHUNK_WEIGHTS = {"long": 3.0, "summary": 2.0, "short": 1.0}
#
# # === AUTH ===
# credentials = service_account.Credentials.from_service_account_file(
#     SERVICE_ACCOUNT_FILE,
#     scopes=["https://www.googleapis.com/auth/bigquery"]
# )
# bq = bigquery.Client(credentials=credentials, project="enduring-brace-451209-q3")
# es = Elasticsearch("http://localhost:9200")
# model = SentenceTransformer("bert-base-nli-mean-tokens")
#
# # === USER INPUT ===
# query = input("🎬 Describe the kind of movie you're looking for: ").strip()
# query_vector = model.encode(query).tolist()
#
# # === HYBRID SEARCH ===
# search_body = {
#     "size": TOP_K,
#     "_source": ["movie_id", "chunk_id", "text", "type", "vector"],
#     "query": {
#         "script_score": {
#             "query": {"match": {"text": query}},
#             "script": {
#                 "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
#                 "params": {"query_vector": query_vector}
#             }
#         }
#     }
# }
# res = es.search(index=ES_INDEX, body=search_body)
#
# # === GROUP CHUNKS ===
# chunk_vectors = defaultdict(lambda: defaultdict(list))  # movie_id -> type -> [vectors]
#
# for hit in res["hits"]["hits"]:
#     doc = hit["_source"]
#     movie_id = doc["movie_id"]
#     chunk_type = doc["type"]
#     vector = doc["vector"]
#     chunk_vectors[movie_id][chunk_type].append(vector)
#
# # === WEIGHTED VECTOR AVERAGE ===
# def compute_weighted_vector(movie_chunks):
#     total_vector = np.zeros(len(query_vector))
#     total_weight = 0.0
#
#     for ctype, vectors in movie_chunks.items():
#         if not vectors:
#             continue
#         weight = CHUNK_WEIGHTS.get(ctype, 1.0)
#         mean_vector = np.mean(vectors, axis=0)  # 👈 normalize first
#         total_vector += weight * mean_vector    # 👈 apply weight to mean
#         total_weight += weight
#
#     return total_vector / total_weight if total_weight > 0 else total_vector
#
# movie_vectors = {mid: compute_weighted_vector(v) for mid, v in chunk_vectors.items()}
#
# # === COSINE SIMILARITY TO QUERY ===
# def cosine_sim(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
#
# movie_scores = {mid: cosine_sim(vec, query_vector) for mid, vec in movie_vectors.items()}
# top_movie_ids = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_MOVIES]
# top_movie_ids = [mid for mid, _ in top_movie_ids]
#
# # === FETCH GENRES AND TITLES FROM BIGQUERY ===
# query_str = f"""
# SELECT tconst AS movie_id, movie_title, genres
# FROM {BQ_TABLE}
# WHERE tconst IN UNNEST(@movie_ids)
# """
# job_config = bigquery.QueryJobConfig(
#     query_parameters=[bigquery.ArrayQueryParameter("movie_ids", "STRING", top_movie_ids)]
# )
# genre_df = bq.query(query_str, job_config=job_config).to_dataframe()
# genre_map = dict(zip(genre_df["movie_id"], genre_df["genres"]))
# title_map = dict(zip(genre_df["movie_id"], genre_df["movie_title"]))
#
# # === SORT CHUNKS BY ORDER ===
# def chunk_sort_key(chunk_id):
#     parts = chunk_id.split("-")
#     type_order = {"sh": 0, "short": 0, "summary": 1, "lon": 2, "long": 2}
#     type_val = type_order.get(parts[1], 99)
#     secondary = list(map(int, re.findall(r"\d+", "-".join(parts[2:]))))
#     return (type_val, *secondary)
#
# # === DISPLAY RESULTS ===
# print("\n🎯 Top Movie Recommendations:\n")
#
# for movie_id in top_movie_ids:
#     # ✅ Retrieve all chunks for this movie
#     full_chunk_res = es.search(index=ES_INDEX, body={
#         "size": 1000,
#         "_source": ["chunk_id", "text", "type"],
#         "query": {"term": {"movie_id.keyword": movie_id}}
#     })
#     full_chunks = [
#         {
#             "chunk_id": doc["_source"]["chunk_id"],
#             "text": doc["_source"]["text"],
#             "type": doc["_source"]["type"]
#         }
#         for doc in full_chunk_res["hits"]["hits"]
#     ]
#
#     sorted_chunks = sorted(full_chunks, key=lambda x: chunk_sort_key(x["chunk_id"]))
#     genres = genre_map.get(movie_id, [])
#     title = title_map.get(movie_id, "Unknown Title")
#
#     print(f"🎬 {title} (ID: {movie_id})")
#     print(f"🏷  Genres: {', '.join(genres) if genres else 'Unknown'}")
#     print("📖 Preview:")
#
#     # === SHOW SHORT SYNOPSIS IF AVAILABLE ===
#     for chunk in sorted_chunks[:3]:  # Show top 3 chunks
#          print(f"• ({chunk['type']}) {chunk['text'][:200]}...")