import os
import shutil

# Path to folder containing the JSON files
json_dir = "srsly_final_drama_chunks_1"  # Replace with your folder path
output_dir = "selected_500_drama_ids_jsons"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Get all JSON filenames
all_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

# Step 2: Extract unique movie IDs from filenames
def extract_id(filename):
    return filename.split("-")[0]

unique_ids = []
seen = set()

for file in all_files:
    movie_id = extract_id(file)
    if movie_id not in seen:
        unique_ids.append(movie_id)
        seen.add(movie_id)
    if len(unique_ids) == 500:
        break

# Step 3: Copy all files that start with any of the 500 IDs
for file in all_files:
    movie_id = extract_id(file)
    if movie_id in seen:
        src = os.path.join(json_dir, file)
        dst = os.path.join(output_dir, file)
        shutil.copy(src, dst)