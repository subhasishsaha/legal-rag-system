# Legal RAG System

A Retrieval-Augmented Generation (RAG) system for answering legal questions using a combination of document retrieval, semantic search, and Google Gemini.

## Overview

This project builds a legal research assistant that:

- Loads legal documents from PDFs and web sources.
- Preprocesses and chunks documents.
- Generates embeddings and stores them in ChromaDB.
- Decomposes complex legal questions into smaller sub-queries.
- Retrieves relevant legal context using semantic search.
- Generates grounded answers using Gemini 2.5 Flash.
- Exposes the pipeline through a FastAPI service.

## Architecture

```
User Query
    |
    v
Sub-query Generation
    |
    v
Vector Retrieval (ChromaDB)
    |
    v
Context Aggregation & Deduplication
    |
    v
Gemini 2.5 Flash
    |
    v
Final Legal Answer
```

## Repository Structure

```
legal-rag-system/
│
├── api.py                  # FastAPI application and pipeline initialization
├── main.py                 # Simple client for interacting with deployed API
├── requirements.txt        # Project dependencies
├── Dockerfile              # Container configuration
│
├── chains/
│   └── chains.py           # RAG chain and retrieval workflow
│
├── utils/
│   ├── prompts.py          # Prompt templates
│   ├── preprocessing.py    # Document cleaning and preprocessing
│   ├── document_loaders.py # PDF and web document loading
│   ├── text_splitter.py    # Chunking logic
│   └── vectorstore.py      # ChromaDB creation and loading
│
├── chroma_db/              # Persistent vector database (generated)
└── data/                   # Legal source documents
```

## Key Features

### Multi-Source Legal Knowledge

The system combines:

- Local PDF legal documents.
- Web-based legal resources.
- Persistent vector storage using ChromaDB.

### Query Decomposition

Complex legal questions are automatically broken into focused sub-questions before retrieval, improving recall and answer quality.

### Context Deduplication

Retrieved documents are deduplicated before being passed to the LLM, reducing noise and redundant context.

### FastAPI Backend

Exposes endpoints for:

- `GET /` – API information
- `GET /health` – Service health status
- `POST /ask` – Legal question answering

## Tech Stack

- Python
- FastAPI
- LangChain
- LangGraph
- ChromaDB
- Sentence Transformers
- Google Gemini
- Docker

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/subhasishsaha/legal-rag-system.git
cd legal-rag-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
```

## Running the API

```bash
uvicorn api:app --reload
```

The API will be available at:

```
http://localhost:8000
```

Interactive documentation:

```
http://localhost:8000/docs
```

## Example Request

```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{
  "query": "What are the major Supreme Court judgments of 2025?"
}'
```

Example response:

```json
{
  "answer": "...generated legal response..."
}
```

## Docker

Build the image:

```bash
docker build -t legal-rag-system .
```

Run the container:

```bash
docker run -p 8000:8000 legal-rag-system
```

## Future Improvements

- Citation-aware responses.
- Hybrid retrieval (BM25 + vector search).
- Conversation memory.
- Multi-jurisdiction legal support.
- Evaluation and benchmarking pipeline.
- Streaming responses.

## Disclaimer

This project is intended for educational and research purposes. Generated responses should not be treated as professional legal advice.
