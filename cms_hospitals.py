import os
import json
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from slugify import slugify
from io import StringIO
from pyspark.sql import SparkSession

# --------------------
# CONFIG
# --------------------
CMS_METASTORE_URL = "https://example.com/api/datasets"  # Replace with actual CMS provider API
DATA_DIR = "data"
METADATA_FILE = "metadata.json"
MAX_WORKERS = 5  # parallel downloads

os.makedirs(DATA_DIR, exist_ok=True)

# --------------------
# HELPER FUNCTIONS
# --------------------
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {"last_run": None, "files": {}}

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def to_snake_case(col_name):
    """Convert column name to snake_case"""
    return slugify(col_name, separator="_")

def download_and_process(dataset):
    """Download CSV content into Spark directly and process"""
    dataset_name = dataset["name"]
    dataset_url = dataset["url"]
    last_modified = dataset["last_modified"]

    output_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")

    # Skip if not modified
    if dataset_name in metadata["files"] and metadata["files"][dataset_name] == last_modified:
        print(f"Skipping {dataset_name}, not modified since last run.")
        return

    # Download CSV content
    print(f"Downloading {dataset_name}...")
    resp = requests.get(dataset_url)
    resp.raise_for_status()
    csv_content = resp.text

    # Load CSV into Spark from memory
    csv_buffer = StringIO(csv_content)
    df = spark.read.option("header", True).csv(csv_buffer)

    # Rename columns to snake_case
    for c in df.columns:
        df = df.withColumnRenamed(c, to_snake_case(c))

    # Save CSV
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
    print(f"Saved processed dataset to {output_path}")

    # Update metadata
    metadata["files"][dataset_name] = last_modified

# --------------------
# MAIN JOB
# --------------------
if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("CMS Hospitals Data Job") \
        .getOrCreate()

    metadata = load_metadata()
    last_run = metadata.get("last_run")
    print(f"Last run: {last_run}")

    # 1. Fetch datasets from CMS metastore
    print("Fetching datasets from CMS metastore...")
    response = requests.get(CMS_METASTORE_URL)
    response.raise_for_status()
    all_datasets = response.json()  # [{"name": ..., "url": ..., "theme": ..., "last_modified": ...}, ...]

    # 2. Filter for theme "Hospitals"
    hospital_datasets = [d for d in all_datasets if d.get("theme") == "Hospitals"]

    # 3. Download & process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_and_process, hospital_datasets)

    # 4. Update last run timestamp
    metadata["last_run"] = datetime.utcnow().isoformat()
    save_metadata(metadata)

    print("Job completed successfully.")
    spark.stop()
