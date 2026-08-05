-- Setup script for ticker_news_documents table
-- Run this manually in your Lakebase Postgres database before running the notebook

-- Create the news documents table
CREATE TABLE IF NOT EXISTS ticker_news_documents (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    author TEXT,
    article_url TEXT,
    publisher_name TEXT,
    keywords JSONB,
    sentiment TEXT,
    sentiment_reasoning TEXT,
    published_utc TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create index for ticker lookups
CREATE INDEX IF NOT EXISTS idx_ticker_news_documents_ticker 
ON ticker_news_documents (ticker);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'ticker_news_documents'
ORDER BY ordinal_position;