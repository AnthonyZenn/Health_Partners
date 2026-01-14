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
CMS_METASTORE_URL = "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items"
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
    """Download CSV into Spark directly and process"""
    dataset_name = dataset.get("identifier", dataset.get("title", "unknown_dataset"))
    distributions = dataset.get("distribution", [])
    if not distributions:
        print(f"Skipping {dataset_name}: no distributions found.")
        return

    # Pick first CSV download URL
    dataset_url = None
    for dist in distributions:
        if dist.get("mediaType") == "text/csv" and dist.get("downloadURL"):
            dataset_url = dist["downloadURL"]
            break

    if not dataset_url:
        print(f"Skipping {dataset_name}: no CSV download URL found.")
        return

    last_modified = dataset.get("modified")  # e.g., "2025-10-14"
    output_path = os.path.join(DATA_DIR, dataset_name.replace(" ", "_"))

    # Skip if not modified
    if dataset_name in metadata["files"] and metadata["files"][dataset_name] == last_modified:
        print(f"Skipping {dataset_name}, not modified since last run.")
        return

    # Download CSV content
    print(f"Downloading {dataset_name} from {dataset_url} ...")
    resp = requests.get(dataset_url)
    resp.raise_for_status()
    csv_content = resp.text

    # Load CSV into Spark from memory
    csv_buffer = StringIO(csv_content)
    df = spark.read.option("header", True).csv(csv_buffer)

    # Rename columns to snake_case
    for c in df.columns:
        df = df.withColumnRenamed(c, to_snake_case(c))

    # Save CSV locally
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
    print(f"Saved processed dataset to {output_path}")

    # Update metadata
    metadata["files"][dataset_name] = last_modified

# --------------------
# MAIN JOB
# --------------------
if __name__ == "__main__":
    # Initialize Spark in local mode
    spark = SparkSession.builder \
        .appName("CMS Hospitals Data Job") \
        .master("local[*]") \   # <-- local execution on all cores
        .getOrCreate()

    metadata = load_metadata()
    last_run = metadata.get("last_run")
    print(f"Last run: {last_run}")

    # Fetch datasets from CMS
    print("Fetching datasets from CMS metastore...")
    response = requests.get(CMS_METASTORE_URL)
    response.raise_for_status()
    all_datasets = response.json()  # list of dataset metadata

    # Filter datasets for theme "Hospitals" (theme is a list)
    hospital_datasets = [
        d for d in all_datasets
        if "theme" in d and isinstance(d["theme"], list) and "Hospitals" in d["theme"]
    ]

    print(f"Found {len(hospital_datasets)} hospital datasets to process.")

    # Download & process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_and_process, hospital_datasets)

    # Update last run timestamp
    metadata["last_run"] = datetime.utcnow().isoformat()
    save_metadata(metadata)

    print("Job completed successfully.")
    spark.stop()
