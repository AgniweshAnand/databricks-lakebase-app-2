-- Setup script for ticker_news_embeddings table
-- Run this manually in your Lakebase Postgres database before running the notebook
-- Replace {{EMBEDDING_DIM}} with your model's dimension (e.g., 384 for all-MiniLM-L6-v2)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the embeddings table
-- IMPORTANT: Replace {{EMBEDDING_DIM}} below with the correct dimension for your model:
--   - sentence-transformers/all-MiniLM-L6-v2: 384
--   - sentence-transformers/all-mpnet-base-v2: 768
--   - BAAI/bge-small-en-v1.5: 384
--   - BAAI/bge-base-en-v1.5: 768
--   - BAAI/bge-large-en-v1.5: 1024
CREATE TABLE IF NOT EXISTS ticker_news_embeddings (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    published_utc TIMESTAMPTZ,
    embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_ticker_news_embeddings_embedding
ON ticker_news_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'ticker_news_embeddings'
ORDER BY ordinal_position;