# Enterprise RAG Assistant

A Retrieval-Augmented Generation (RAG) application built with:

- Streamlit
- Anthropic Claude
- Voyage AI Embeddings
- Pinecone

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag-demo
```

## Ingest Documents

Place PDFs into:

```text
data/raw/
```

Run:

```bash
python -m scripts.ingest_folder
```

## Run Application

```bash
python -m streamlit run app.py
```