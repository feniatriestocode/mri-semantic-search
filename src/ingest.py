# src/ingest.py
from pathlib import Path
from tqdm import tqdm
from psycopg2.extras import execute_values

from src.db import get_connection
from src.embedder import BiomedClipEmbedder

DATA_DIR = Path("data/archive/Training")
BATCH_SIZE = 32 

def collect_image_metadata():
    """Collect image metadata."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    items = []
    
    # Διατρέχει: data/<class_name>/<image_file>
    for image_path in DATA_DIR.glob("*/*.*"):
        if image_path.suffix.lower() in valid_extensions:
            items.append({
                "path": str(image_path),
                "image_name": image_path.name,
                "category": image_path.parent.name
            })
    return items

def run_ingestion():
    # 1. Image Collection
    images = collect_image_metadata()
    total_images = len(images)
    print(f"{total_images} images collected.")
    if total_images == 0:
        print("Warning: No images found in the data directory.")
        return

    # 2. Initialization of Embedder and DB Connection
    print("Loading BiomedCLIP model...")
    embedder = BiomedClipEmbedder()
    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO image_items (image_name, category, embedding)
        VALUES %s;
    """

    # 3. Edit in batches (extraction + storage)
    print("Starting Ingestion...")
    for i in tqdm(range(0, total_images, BATCH_SIZE), desc="Ingesting"):
        batch_items = images[i : i + BATCH_SIZE]
        batch_paths = [item["path"] for item in batch_items]

        # Pass through the model -> list of 512-d vectors
        embeddings = embedder.embed_batch(batch_paths)

        # Prepare tuples for SQL Insertion: (image_name, category, embedding)
        records = [
            (item["image_name"], item["category"], emb)
            for item, emb in zip(batch_items, embeddings)
        ]

        # Speedy hehe Bulk Insert w/ execute_values
        execute_values(cursor, insert_sql, records, template="(%s, %s, %s::vector)")
        conn.commit()

    cursor.close()
    conn.close()
    print("Ingestion done!")

if __name__ == "__main__":
    run_ingestion()