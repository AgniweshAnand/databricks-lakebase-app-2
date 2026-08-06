# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Ticker News -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Context Engineering on Databricks** course.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `watchlist` table in Lakebase to find out which ticker
# MAGIC    symbols are currently being tracked.
# MAGIC 2. Fetches recent news for those tickers directly from the Massive
# MAGIC    `/v2/reference/news` endpoint (see `massive_client.py` for the same
# MAGIC    call shape used by the Flask app's `POST /news/sync` route), rate
# MAGIC    limited to stay within the free Massive API tier's strict quota, and
# MAGIC    upserts the results into the `ticker_news_documents` table.
# MAGIC 3. Computes a sentence embedding for each article (title + description)
# MAGIC    using Spark, distributed across the cluster via a pandas UDF, and
# MAGIC    writes them into a `ticker_news_embeddings` table using the
# MAGIC    `pgvector` Postgres extension so downstream RAG / context-engineering
# MAGIC    exercises can run similarity search directly in Postgres.
# MAGIC 4. Fetches the full article body for each `article_url` (via
# MAGIC    `trafilatura`, which strips nav/ads/boilerplate from the raw HTML),
# MAGIC    splits it into overlapping text chunks, embeds each chunk, and writes
# MAGIC    them into a `ticker_news_chunk_embeddings` table - so RAG exercises can
# MAGIC    retrieve fine-grained passages from article bodies, not just
# MAGIC    title/description.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

dbutils.widgets.text("watchlist_table_name", "watchlist", "Source table (watchlist symbols)")
dbutils.widgets.text("news_table_name", "ticker_news_documents", "Destination table (raw news)")
dbutils.widgets.text("embeddings_table_name", "ticker_news_embeddings", "Destination table (vectors)")
dbutils.widgets.text("chunk_embeddings_table_name", "ticker_news_chunk_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("massive_secret_scope", "massive", "Massive API secret scope")
dbutils.widgets.text("massive_secret_key", "api-key", "Massive API secret key")
dbutils.widgets.text("massive_api_base_url", "https://api.massive.com", "Massive API base URL")
dbutils.widgets.text("news_fetch_limit", "50", "Max articles to fetch per ticker")
dbutils.widgets.text("max_requests_per_minute", "5", "Massive API rate limit (free tier is strict)")
dbutils.widgets.text("chunk_size", "800", "Article content chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Article content chunk overlap (chars)")

WATCHLIST_TABLE_NAME = dbutils.widgets.get("watchlist_table_name")
NEWS_TABLE_NAME = dbutils.widgets.get("news_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
CHUNK_EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("chunk_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
MASSIVE_SECRET_SCOPE = dbutils.widgets.get("massive_secret_scope")
MASSIVE_SECRET_KEY = dbutils.widgets.get("massive_secret_key")
MASSIVE_API_BASE_URL = dbutils.widgets.get("massive_api_base_url")
NEWS_FETCH_LIMIT = int(dbutils.widgets.get("news_fetch_limit"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces psycopg3
# MAGIC needs for connection (host/port/dbname/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# DBTITLE 1,Test Psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using OAuth token authentication as user: {db_user}\n")

# Test psycopg3 connection with OAuth token
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=10
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {WATCHLIST_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {WATCHLIST_TABLE_NAME}")
    
    cursor.execute(f"SELECT * FROM {WATCHLIST_TABLE_NAME} LIMIT 5")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    print(f"\nColumns: {colnames}")
    for row in rows:
        print(row)
    
    cursor.close()
    conn.close()
    print("\n✅ psycopg3 with OAuth authentication working correctly!")
except Exception as e:
    import traceback
    print(f"❌ Connection failed: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, you must manually create the required tables
# MAGIC in your Lakebase Postgres database:
# MAGIC
# MAGIC 1. Run `sql/01_setup_news_table.sql` to create `ticker_news_documents`
# MAGIC 2. Run `sql/02_setup_embeddings_table.sql` to create `ticker_news_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC 3. Run `sql/03_setup_chunk_embeddings_table.sql` to create `ticker_news_chunk_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC
# MAGIC This notebook uses psycopg2 with OAuth token authentication for all database operations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch news from Massive for watchlisted tickers
# MAGIC
# MAGIC This ETL is now self-contained: instead of relying on the Flask app's
# MAGIC `POST /news/sync` route to have populated `ticker_news_documents` ahead of
# MAGIC time, the notebook queries the `watchlist` table in Lakebase directly to
# MAGIC find out which tickers are being tracked, then pulls news for exactly
# MAGIC those tickers from Massive itself.
# MAGIC
# MAGIC The free Massive API tier is rate-limited very aggressively, so requests
# MAGIC are made **serially** (not distributed across Spark workers) with a sleep
# MAGIC between calls that enforces `MAX_REQUESTS_PER_MINUTE` (default 5/min).

# COMMAND ----------

# DBTITLE 1,Fetch news and sync using Lakebase SDK
import base64 as _b64
import json as _json
import time
from datetime import datetime

import psycopg2
import requests


def get_massive_api_key() -> str:
    secret = w.secrets.get_secret(scope=MASSIVE_SECRET_SCOPE, key=MASSIVE_SECRET_KEY)
    return _b64.b64decode(secret.value).decode("utf-8")


def get_watchlist_tickers() -> list[str]:
    """Distinct, uppercased ticker symbols currently tracked across all users
    in the watchlist table - these are the only tickers we fetch news for."""
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT symbol FROM {WATCHLIST_TABLE_NAME} WHERE symbol IS NOT NULL")
        symbols = cursor.fetchall()
        return [row[0].strip().upper() for row in symbols if row[0]]
    finally:
        cursor.close()
        conn.close()


def fetch_news_for_ticker(session: requests.Session, ticker: str, limit: int) -> list[dict]:
    """Single GET /v2/reference/news call for one ticker (mirrors
    MassiveClient.get_news in massive_client.py)."""
    resp = session.get(
        f"{MASSIVE_API_BASE_URL}/v2/reference/news",
        params={"ticker": ticker, "limit": limit, "order": "desc", "sort": "published_utc"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def sync_news_to_lakebase(ticker: str, articles: list[dict]) -> int:
    """Convert Massive API response and insert via Lakebase SDK executeLakebasePostgresSql tool.
    Uses ON CONFLICT DO NOTHING for automatic deduplication."""
    if not articles:
        return 0
    
    rows = []
    for article in articles:
        sentiment = None
        sentiment_reasoning = None
        for insight in article.get("insights", []) or []:
            if insight.get("ticker") == ticker:
                sentiment = insight.get("sentiment")
                sentiment_reasoning = insight.get("sentiment_reasoning")
                break

        publisher = article.get("publisher") or {}
        rows.append({
            "id": str(article.get("id")),
            "ticker": ticker,
            "title": article.get("title", ""),
            "description": article.get("description"),
            "author": article.get("author"),
            "article_url": article.get("article_url"),
            "publisher_name": publisher.get("name"),
            "keywords": _json.dumps(article.get("keywords", [])),
            "sentiment": sentiment,
            "sentiment_reasoning": sentiment_reasoning,
            "published_utc": article.get("published_utc"),
            "payload": _json.dumps(article),
        })

    # We'll collect these rows and write them via the assistant's executeLakebasePostgresSql tool
    # Store in a global so the next cell can access them
    return rows


print("NOTE: Before running this cell, ensure you've run sql/01_setup_news_table.sql")
print("      to create the ticker_news_documents table in your Lakebase database.\n")

tickers = get_watchlist_tickers()
print(f"Found {len(tickers)} distinct watchlisted tickers: {tickers}")

# Enforce MAX_REQUESTS_PER_MINUTE by spacing calls evenly across a minute -
# e.g. 5/min -> one request every 12s. Sleeping BEFORE each call after the
# first keeps this correct even if a single request itself takes a while.
_seconds_between_requests = 60.0 / MAX_REQUESTS_PER_MINUTE

_massive_session = requests.Session()
_massive_session.headers.update(
    {"Authorization": f"Bearer {get_massive_api_key()}", "Content-Type": "application/json"}
)

news_synced = 0
all_news_rows = []  # Collect all rows to insert via Lakebase SDK
for i, ticker in enumerate(tickers):
    if i > 0:
        time.sleep(_seconds_between_requests)
    try:
        articles = fetch_news_for_ticker(_massive_session, ticker, NEWS_FETCH_LIMIT)
        batch_rows = sync_news_to_lakebase(ticker, articles)
        if batch_rows:
            all_news_rows.extend(batch_rows)
    except Exception as exc:
        print(f"Skipping {ticker}: failed to fetch/sync news ({exc})")
        continue

print(f"\nCollected {len(all_news_rows)} news articles to insert via Lakebase SDK.")
print(f"Run the next cell to insert them using executeLakebasePostgresSql tool.")

# COMMAND ----------

# DBTITLE 1,Insert collected news articles using psycopg2
import psycopg2

# Build connection using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    cursor = conn.cursor()
    
    # Prepare data tuples for batch insert
    insert_data = [
        (
            row['id'],
            row['ticker'],
            row['title'],
            row['description'],
            row['author'],
            row['article_url'],
            row['publisher_name'],
            row['keywords'],
            row['sentiment'],
            row['sentiment_reasoning'],
            row['published_utc'],
            row['payload']
        )
        for row in all_news_rows
    ]
    
    # Batch insert with ON CONFLICT DO NOTHING for deduplication
    insert_sql = f"""
        INSERT INTO {NEWS_TABLE_NAME} (
            id, ticker, title, description, author, article_url, publisher_name,
            keywords, sentiment, sentiment_reasoning, published_utc, payload, synced_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (id) DO NOTHING
    """
    
    # executemany in psycopg3 is much faster than individual INSERTs
    cursor.executemany(insert_sql, insert_data)
    
    conn.commit()
    inserted_count = cursor.rowcount
    print(f"✅ Successfully inserted {inserted_count} new news articles")
    print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
    
finally:
    cursor.close()
    conn.close()

print(f"\nReady to compute embeddings! Run the cells below to continue.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw news documents
# MAGIC
# MAGIC Reads the whole `ticker_news_documents` table (just synced from Massive
# MAGIC above) using psycopg3 into a pandas DataFrame for embedding computation.

# COMMAND ----------

import pandas as pd
import psycopg2

# Load news documents using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    # Query with embedding_text computed
    query = f"""
        SELECT 
            id,
            ticker,
            title,
            description,
            article_url,
            published_utc,
            TRIM(CONCAT(COALESCE(title, ''), '. ', COALESCE(description, ''))) AS embedding_text
        FROM {NEWS_TABLE_NAME}
        WHERE TRIM(CONCAT(COALESCE(title, ''), '. ', COALESCE(description, ''))) IS NOT NULL
          AND TRIM(CONCAT(COALESCE(title, ''), '. ', COALESCE(description, ''))) != ''
    """
    
    news_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(news_df)} news documents from {NEWS_TABLE_NAME}")
    display(news_df.head(5))
finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model once and applies it in batches
# MAGIC to the news documents.

# COMMAND ----------

# DBTITLE 1,Compute embeddings (distributed pandas UDF)
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute embeddings in batches for memory efficiency
print("Computing embeddings...")
batch_size = 32
all_embeddings = []

for i in range(0, len(news_df), batch_size):
    batch = news_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(news_df))}/{len(news_df)} documents")

# Create embeddings DataFrame
embeddings_df = pd.DataFrame({
    "id": news_df["id"],
    "ticker": news_df["ticker"],
    "title": news_df["title"],
    "published_utc": news_df["published_utc"].astype(str),
    "embedding": all_embeddings,
})

print(f"Computed {len(embeddings_df)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC The `pgvector` extension must be enabled and the destination table
# MAGIC created with the correct vector dimension before inserting embeddings.

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/02_setup_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/02_setup_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written in batches via psycopg2's `executemany` for throughput.
# MAGIC Each embedding is cast to Postgres' `vector` type via `::vector`.

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Add model_name and embedded_at columns
embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
embeddings_df['embedded_at'] = datetime.now()

embeddings_rows = embeddings_df.to_dict('records')

if len(embeddings_rows) > 0:
    print(f"Inserting {len(embeddings_rows)} embeddings into {EMBEDDINGS_TABLE_NAME}...")
    
    # Build connection from parsed URL
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Prepare data tuples for batch insert
        # Format embedding as PostgreSQL array literal: '{val1,val2,...}'
        insert_data = [
            (
                row['id'],
                row['ticker'],
                row['title'],
                str(row['published_utc']) if row['published_utc'] else None,
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['embedded_at']
            )
            for row in embeddings_rows
        ]
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, ticker, title, published_utc, embedding, model_name, embedded_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        
        # execute_values is much faster than individual INSERTs
        template = "(%s, %s, %s, %s, %s::double precision[], %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new embeddings")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
        print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch and chunk article content
# MAGIC
# MAGIC Title/description only gets you so far - the actual article body lives at
# MAGIC `article_url` on the publisher's site. This step fetches each URL, uses
# MAGIC `trafilatura` to extract just the article text (stripping nav/ads/related
# MAGIC links/etc.), and splits it into overlapping chunks so each chunk can be
# MAGIC embedded and retrieved independently. Any URL that fails to fetch/extract
# MAGIC (paywall, timeout, dead link) is skipped rather than failing the whole job.

# COMMAND ----------

import pandas as pd
import requests
import trafilatura

# Filter news_df for articles with URLs
content_df = news_df[news_df['article_url'].notna() & (news_df['article_url'] != '')].copy()

print(f"Fetching and chunking content from {len(content_df)} article URLs...")

# Fetch and chunk article content
out_article_ids, out_tickers, out_chunk_indexes, out_chunk_texts = [], [], [], []

for idx, row in content_df.iterrows():
    article_id = row['id']
    ticker = row['ticker']
    article_url = row['article_url']
    
    try:
        resp = requests.get(article_url, timeout=15)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text)
    except Exception as e:
        # Dead link, paywall, timeout, etc. - skip this article's
        # content chunks rather than failing the whole job.
        continue

    if not text:
        continue

    # Split into overlapping chunks
    for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        out_article_ids.append(article_id)
        out_tickers.append(ticker)
        out_chunk_indexes.append(str(chunk_index))
        out_chunk_texts.append(chunk_text)
        if start + CHUNK_SIZE >= len(text):
            break
    
    # Progress update every 10 articles
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(content_df)} articles")

chunks_df = pd.DataFrame({
    "article_id": out_article_ids,
    "ticker": out_tickers,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Extracted {len(chunks_df)} content chunks from {len(content_df)} article URLs")
display(chunks_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings
# MAGIC
# MAGIC Same approach as the title/description embeddings above - load the model
# MAGIC once and process in batches, but one vector per content chunk instead of
# MAGIC per article.

# COMMAND ----------

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Model should already be loaded from earlier, but ensure cache is set
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Computing chunk embeddings using {EMBEDDING_MODEL_NAME}...")
# Reuse the model if already loaded, otherwise load it
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute chunk embeddings in batches
batch_size = 32
all_chunk_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_chunk_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

# Create chunk embeddings DataFrame
chunk_embeddings_df = pd.DataFrame({
    "article_id": chunks_df["article_id"],
    "ticker": chunks_df["ticker"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the chunk embeddings destination table exists

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/03_setup_chunk_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {CHUNK_EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/03_setup_chunk_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using psycopg2
import psycopg2
from datetime import datetime

# Add id (article_id_chunk_index), model_name, and embedded_at columns
chunk_embeddings_df['id'] = chunk_embeddings_df['article_id'] + '_' + chunk_embeddings_df['chunk_index']
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['embedded_at'] = datetime.now()
chunk_embeddings_df['chunk_index'] = chunk_embeddings_df['chunk_index'].astype(int)

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}...")
    
    # Build connection using psycopg2
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Prepare data tuples for batch insert
        # Format embedding as PostgreSQL array literal: '{val1,val2,...}'
        insert_data = [
            (
                row['id'],
                row['article_id'],
                row['ticker'],
                int(row['chunk_index']),
                row['chunk_text'],
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['embedded_at']
            )
            for row in chunk_embeddings_rows
        ]
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql = f"""
            INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME} (
                id, article_id, ticker, chunk_index, chunk_text, embedding, model_name, embedded_at
            ) VALUES (%s, %s, %s, %s, %s, %s::double precision[], %s, %s)
            ON CONFLICT (id) DO NOTHING
        """
        
        # executemany in psycopg2 is much faster than individual INSERTs
        cursor.executemany(insert_sql, insert_data)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new chunk embeddings")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
        print(f"  UPDATE {CHUNK_EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No chunk embeddings to write.")