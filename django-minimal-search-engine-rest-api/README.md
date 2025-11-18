# Search Engine API - Minimal Django REST Implementation

## Features

- **Semantic Search**: Uses [sentence transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for document and query embedding
- **REST API**: Three endpoints - `/status/`, `/query/` and `/health/`
- **Evaluation Metrics**: Calculates Precision@5 (P@5) for search quality
- **Caching**: LRU cache for query embeddings

## Requirements

- Python 3.10+
- Django 5.1
- Django REST Framework 3.15.2
- sentence-transformers 3.0.1
- numpy 1.26.4
- scikit-learn 1.5.1

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone
cd django-minimal-rest-api

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Complete setup (download dataset, migrate DB, index documents)
make setup

# Run the server
make run
```

The API will be available at `http://127.0.0.1:8000/`

---

### Option 2: Manual Setup

#### 1. Clone and Install

```bash
git clone
cd django-minimal-rest-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Download Dataset

```bash
# Using Make (recommended)
make download-data

# Or manually
mkdir -p data && cd data && \
  wget https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz && \
  tar -xzf nfcorpus.tar.gz && \
  mv nfcorpus/train.docs . && \
  mv nfcorpus/train.titles.queries . && \
  mv nfcorpus/train.3-2-1.qrel . && \
  rm nfcorpus.tar.gz && rm -rf nfcorpus && cd ..
```

#### 3. Initialize Database and Index Documents

```bash
# Run migrations
make migrate
# Or: python manage.py migrate

# Index documents (this may take a few minutes)
make index
# Or: python manage.py init --clear
```

#### 4. Run the Server

```bash
make run
# Or: python manage.py runserver
```

---

## Docker Setup

### Quick Start with Docker Compose

```bash
# Clone repository
git clone
cd django-minimal-rest-api
make download-data

# Start services  & build image
docker-compose up -d

# View logs
docker-compose logs -f web

# Access API at http://localhost:8000/
```

### Manual Docker Setup

```bash
# Build the image
docker build -t search-engine-api .

# Run the container
docker run -d \
  --name search-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  search-engine-api

# View logs
docker logs -f search-api
```

### Stop and Clean Up

```bash
# Using Docker Compose
docker-compose down

# Using Docker directly
docker stop search-api
docker rm search-api
```

---

## Makefile Commands

```bash
make help           # Show all available commands
make download-data  # Download and setup NFCorpus dataset
make clean-data     # Remove dataset files
make migrate        # Run database migrations
make index          # Index documents (clears existing data)
make test           # Run test suite
make run            # Start development server
make setup          # Full setup (download + migrate + index)
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/api/health/
```

**Response:**

```json
{
  "status": "healthy"
}
```

### System Status

```bash
curl -X POST http://localhost:8000/api/status/
```

**Response:**

```json
{
  "num_of_indexed_items": 3633,
  "num_of_queries_in_qrels": 2594
}
```

### Query Search

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "PLAIN-831",
    "query_text": "cardiovascular disease"
  }'
```

**Response:**

```json
{
  "top_docs": [
    "MED-2590",
    "MED-1634",
    "MED-1409",
    "MED-3099",
    "MED-2530",
    "MED-4247",
    "MED-4616",
    "MED-4891",
    "MED-3439",
    "MED-2370"
  ],
  "p5": 0.6
}
```

---

## Development

### Run Tests

```bash
# Run all tests
make test
# or
python manage.py test
```

### Test api

```bash
./test_api.sh
```
