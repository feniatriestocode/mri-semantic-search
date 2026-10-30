# MRI Semantic Search

Semantic search over brain MRI images using **[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)** embeddings stored in **PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)**.

Images are encoded into 512-dimensional vectors and stored alongside their class label. Because BiomedCLIP embeds images and text into the same space, the same database can be used for image-to-image similarity search and text-to-image search.

> **Status:** work in progress. The database setup, embedding model wrapper and ingestion pipeline are implemented. Vector indexing, query helpers and benchmarking are planned (see [Roadmap](#roadmap)).

## How it works

```
 MRI images (data/<class>/<file>)
          │
          ▼
 BiomedCLIP image encoder ──► 512-d, L2-normalised vector
          │
          ▼
 PostgreSQL + pgvector  (table: image_items)
          ▲
          │  cosine distance
 BiomedCLIP text / image encoder ◄── query
```

- Embeddings are **L2-normalised**, so cosine similarity is the intended metric.
- Ingestion runs in batches of 32 and bulk-inserts with `execute_values`.
- The model runs on CUDA when available and falls back to CPU.

## Repository structure

```
.
├── docker-compose.yml     # pgvector/pgvector:pg16 database
├── pyproject.toml         # package metadata and dependencies
├── sql/
│   ├── init.sql           # enables pgvector, creates image_items
│   ├── create_index.sql   # (planned) vector index
│   └── queries.sql        # (planned) example queries
└── src/
    ├── db.py              # psycopg2 connection with pgvector registered
    ├── embedder.py        # BiomedClipEmbedder (image + text embeddings)
    ├── ingest.py          # embeds images and loads them into the DB
    └── benchmark.py       # (planned) search benchmarks
```

## Database schema

```sql
CREATE TABLE image_items (
    id         SERIAL PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,   -- image filename
    category   VARCHAR(100) NOT NULL,   -- class label (parent folder name)
    embedding  vector(512)  NOT NULL,   -- BiomedCLIP embedding
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

## Getting started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- A GPU is optional but speeds up ingestion considerably

### 1. Clone and install

```bash
git clone https://github.com/feniatriestocode/mri-semantic-search.git
cd mri-semantic-search

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Optional development extras (JupyterLab, ipywidgets):

```bash
pip install -e ".[dev]"
```

### 2. Start the database

```bash
docker compose up -d
```

This starts PostgreSQL 16 with pgvector on `localhost:5432` (database `vectordb`) and runs `sql/init.sql` on first start to create the schema.

> **Note:** `init.sql` runs only when the data volume is first created, and it drops and recreates `image_items`. To reset the database, run `docker compose down -v` and start it again.

### 3. Add your dataset

Place the images under `data/archive/Training/`, with one sub-folder per class:

```
data/archive/Training/
├── <class_a>/
│   ├── img_001.jpg
│   └── ...
├── <class_b>/
│   └── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`. The sub-folder name is stored as the image's `category`. The `data/` directory is git-ignored. If your dataset lives elsewhere, change `DATA_DIR` in `src/ingest.py`.

### 4. Run ingestion

From the repository root:

```bash
python -m src.ingest
```

The first run downloads BiomedCLIP from the Hugging Face Hub. A progress bar shows batch progress.

## Usage

### Generating embeddings

```python
from src.embedder import BiomedClipEmbedder

embedder = BiomedClipEmbedder()

img_vec  = embedder.embed_image("path/to/scan.jpg")   # list[float], length 512
text_vec = embedder.embed_text("MRI of a brain tumor")  # list[float], length 512
```

### Example similarity query

The `sql/queries.sql` file is not populated yet. As an illustration, a text-to-image search could look like this (`<=>` is pgvector's cosine distance operator):

```python
from src.db import get_connection
from src.embedder import BiomedClipEmbedder

embedder = BiomedClipEmbedder()
query_vec = embedder.embed_text("MRI of a brain tumor")

conn = get_connection()
cur = conn.cursor()
cur.execute(
    """
    SELECT image_name, category, embedding <=> %s::vector AS distance
    FROM image_items
    ORDER BY distance
    LIMIT 5;
    """,
    (query_vec,),
)
for row in cur.fetchall():
    print(row)
```

Passing `embedder.embed_image(...)` instead of `embed_text(...)` gives image-to-image search.

## Configuration

Database settings are defined in `src/db.py` (`DB_CONFIG`) and match `docker-compose.yml`:

| Setting  | Value       |
|----------|-------------|
| Host     | `localhost` |
| Port     | `5432`      |
| Database | `vectordb`  |
| User     | `postgres`  |
| Password | `password`  |

These are local-development defaults. Change them before exposing the database anywhere beyond your machine.

## Tech stack

- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) via [open_clip](https://github.com/mlfoundations/open_clip)
- PyTorch / torchvision
- PostgreSQL 16 + pgvector
- psycopg2, NumPy, pandas, Pillow, tqdm, matplotlib

## Roadmap

- [x] Database schema and Docker setup
- [x] BiomedCLIP embedder (image, batch and text)
- [x] Batch ingestion pipeline
- [x] Vector index (HNSW / IVFFlat) in `sql/create_index.sql`
- [x] Reusable search queries in `sql/queries.sql`
- [ ] Benchmarking (recall and latency, exact vs. indexed search) in `src/benchmark.py`
- [ ] Search interface / demo notebook

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and must not be used for diagnosis or clinical decision-making.

## License

No license has been specified yet. Add a `LICENSE` file to clarify how others may use this code.

## Acknowledgements

- Microsoft Research for [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- The [pgvector](https://github.com/pgvector/pgvector) project
