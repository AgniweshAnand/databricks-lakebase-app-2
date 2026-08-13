# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
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

# MAGIC %pip install sentence-transformers torch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install sentence-transformers torch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers requests pandas

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

dbutils.widgets.text("weather_documents_table", "weather_documents", "Source table (weather documents)")
dbutils.widgets.text("weather_embeddings_table", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")

WEATHER_DOCS_TABLE = dbutils.widgets.get("weather_documents_table")
WEATHER_EMBEDDINGS_TABLE = dbutils.widgets.get("weather_embeddings_table")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Switch on model name to ensure vector dimensions match pgvector's VECTOR(N) type
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

# Initialize WorkspaceClient
w = WorkspaceClient()

def get_lakebase_url() -> str:
    """Fetch and decode Lakebase connection string from Databricks Secret Scope."""
    # SDK method returns base64 encoded bytes string in .value
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")

# Retrieve and parse connection URL
lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection fields with fallbacks
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/') if parsed.path else "databricks_postgres"
db_user = parsed.username
db_password = parsed.password

print("Connection details parsed successfully:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print("  Auth: Password-based authentication extracted from secret.")

# COMMAND ----------

# DBTITLE 1,Test Psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using password credentials as user: {db_user}\n")

# Test psycopg2 connection to Lakebase
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
    
    # Query weather_documents table count
    cursor.execute(f"SELECT COUNT(*) FROM {WEATHER_DOCS_TABLE}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {WEATHER_DOCS_TABLE}")
    
    # Query sample rows
    cursor.execute(f"SELECT id, location, source_type, headline FROM {WEATHER_DOCS_TABLE} LIMIT 5")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    print(f"\nColumns: {colnames}")
    for row in rows:
        print(row)
    
    cursor.close()
    conn.close()
    print("\n✅ psycopg2 connection working correctly!")
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

# Skipped for Weather Pipeline
# Weather documents are harvested and synced directly to Lakebase via POST /weather/sync in app.py.
print("Weather documents are synced directly via POST /weather/sync. Proceeding to the embedding pipeline.")


# COMMAND ----------

# DBTITLE 1,Insert collected news articles using psycopg2
# Skipped for Weather Pipeline
# News article insertion is no longer needed. Raw weather data is stored via POST /weather/sync.
print("Skipping raw news insertion. Ready to load weather documents for chunking and embedding.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw news documents
# MAGIC
# MAGIC Reads the whole `ticker_news_documents` table (just synced from Massive
# MAGIC above) using psycopg3 into a pandas DataFrame for embedding computation.

# COMMAND ----------

import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    query = f"""
        SELECT 
            d.id,
            d.location,
            d.headline,
            d.narrative_text AS embedding_text
        FROM {WEATHER_DOCS_TABLE} d
        LEFT JOIN {WEATHER_EMBEDDINGS_TABLE} e ON d.id = e.document_id
        WHERE e.id IS NULL 
          AND d.narrative_text IS NOT NULL 
          AND d.narrative_text != ''
    """
    
    docs_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(docs_df)} unembedded weather documents from {WEATHER_DOCS_TABLE}")
    
    # Check if empty before calling display() to avoid CANNOT_INFER_EMPTY_SCHEMA
    if not docs_df.empty:
        display(docs_df.head(5))
    else:
        print("No new unembedded documents found.")

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

if docs_df.empty:
    print("No documents found in docs_df. Skipping embedding calculation.")
    embeddings_df = pd.DataFrame(columns=["id", "location", "headline", "embedding"])
else:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    # Compute embeddings for raw document text in batches
    print("Computing embeddings...")
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(docs_df), batch_size):
        batch = docs_df.iloc[i:i+batch_size]
        vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
        all_embeddings.extend(vectors.tolist())
        if (i + batch_size) % 128 == 0 or (i + batch_size) >= len(docs_df):
            print(f"   Processed {min(i + batch_size, len(docs_df))}/{len(docs_df)} documents")

    # Create embeddings DataFrame
    embeddings_df = pd.DataFrame({
        "id": docs_df["id"].values,
        "location": docs_df["location"].values,
        "headline": docs_df["headline"].values,
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
print(f"Table name: {WEATHER_EMBEDDINGS_TABLE}")
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

# Add model_name and created_at columns
embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
embeddings_df['created_at'] = datetime.now()

embeddings_rows = embeddings_df.to_dict('records')

if len(embeddings_rows) > 0:
    print(f"Inserting {len(embeddings_rows)} chunk embeddings into {WEATHER_EMBEDDINGS_TABLE}...")
    
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
        # Primary key formatted as: <document_id>_chunk_<index>
        insert_data = [
            (
                f"{row['id']}_chunk_0",
                row['id'],
                0,
                row['headline'],
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['created_at']
            )
            for row in embeddings_rows
        ]
        
        # Batch insert with direct pgvector casting (%s::vector)
        insert_sql = f"""
            INSERT INTO {WEATHER_EMBEDDINGS_TABLE} (
                id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        
        template = "(%s, %s, %s, %s, %s::vector, %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new vector embeddings into {WEATHER_EMBEDDINGS_TABLE}.")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        
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

# Extract and chunk narrative text directly from weather documents
out_doc_ids, out_chunk_indexes, out_chunk_texts = [], [], []

for idx, row in docs_df.iterrows():
    doc_id = row['id']
    text = row['embedding_text'].strip() if row['embedding_text'] else ""
    
    if not text:
        continue

    # Split narrative text into overlapping sliding-window chunks
    step = CHUNK_SIZE - CHUNK_OVERLAP if CHUNK_SIZE > CHUNK_OVERLAP else CHUNK_SIZE
    for chunk_index, start in enumerate(range(0, len(text), step)):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
            
        out_doc_ids.append(doc_id)
        out_chunk_indexes.append(chunk_index)
        out_chunk_texts.append(chunk_text)
        
        if start + CHUNK_SIZE >= len(text):
            break

# Explicitly assign columns to avoid schema inference issues on empty datasets
chunks_df = pd.DataFrame({
    "document_id": pd.Series(out_doc_ids, dtype="str"),
    "chunk_index": pd.Series(out_chunk_indexes, dtype="int"),
    "chunk_text": pd.Series(out_chunk_texts, dtype="str"),
})

print(f"Extracted {len(chunks_df)} content chunks from {len(docs_df)} weather documents.")

# Safely render output without triggering CANNOT_INFER_EMPTY_SCHEMA
if not chunks_df.empty:
    display(chunks_df.head(5))
else:
    print("No text chunks generated (docs_df was empty).")

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

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Computing weather chunk embeddings using {EMBEDDING_MODEL_NAME}...")

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
        print(f"   Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

# Create chunk embeddings DataFrame mapped to weather document schema
chunk_embeddings_df = pd.DataFrame({
    "document_id": chunks_df["document_id"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} weather chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the chunk embeddings destination table exists

# COMMAND ----------

# Before running the cells below, ensure you've manually run your SQL setup script in Lakebase.
# Note: For weather documents, all chunk embeddings are stored directly in weather_embeddings.
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {WEATHER_EMBEDDINGS_TABLE}")
print("\nEnsure the weather_embeddings table is created in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using psycopg2
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Build composite ID (document_id_chunk_index) and add metadata columns
chunk_embeddings_df['id'] = chunk_embeddings_df['document_id'] + '_chunk_' + chunk_embeddings_df['chunk_index'].astype(str)
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['created_at'] = datetime.now()
chunk_embeddings_df['chunk_index'] = chunk_embeddings_df['chunk_index'].astype(int)

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {WEATHER_EMBEDDINGS_TABLE}...")
    
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
        # Format embedding as PostgreSQL array literal and cast directly to vector in template
        insert_data = [
            (
                row['id'],
                row['document_id'],
                row['chunk_index'],
                row['chunk_text'],
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['created_at']
            )
            for row in chunk_embeddings_rows
        ]
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql = f"""
            INSERT INTO {WEATHER_EMBEDDINGS_TABLE} (
                id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        
        template = "(%s, %s, %s, %s, %s::vector, %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new chunk embeddings into {WEATHER_EMBEDDINGS_TABLE}.")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No chunk embeddings to write.")

# COMMAND ----------

import psycopg2
from sentence_transformers import SentenceTransformer

# 1. Define your test search query
user_query = "flash flood warning and severe heavy rain"

# 2. Convert query into a vector embedding
print(f"Embedding search query: '{user_query}'...")
query_vector = model.encode(user_query).tolist()
query_vector_str = '[' + ','.join(str(float(x)) for x in query_vector) + ']'

# 3. Connect to Lakebase and run vector similarity search (<=> operator)
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
    search_sql = f"""
        SELECT 
            document_id,
            chunk_index,
            chunk_text,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM {WEATHER_EMBEDDINGS_TABLE}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT 5;
    """
    cursor.execute(search_sql, (query_vector_str, query_vector_str))
    results = cursor.fetchall()

    print(f"\n🔍 Top {len(results)} search results for: '{user_query}'")
    print("=" * 65)
    
    if results:
        for doc_id, chunk_idx, text, score in results:
            print(f"Score: {score:.4f} | Doc ID: {doc_id} (Chunk {chunk_idx})")
            print(f"Text:  {text}")
            print("-" * 65)
    else:
        print("No matching vector embeddings found in the database.")

finally:
    cursor.close()
    conn.close()

# COMMAND ----------

import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    df_raw = pd.read_sql_query(f"SELECT COUNT(*) as raw_count FROM {WEATHER_DOCS_TABLE};", conn)
    df_vec = pd.read_sql_query(f"SELECT COUNT(*) as vec_count FROM {WEATHER_EMBEDDINGS_TABLE};", conn)
    print(f"📄 Raw weather_documents count:    {df_raw['raw_count'].iloc[0]}")
    print(f"🔢 Vector weather_embeddings count: {df_vec['vec_count'].iloc[0]}")
finally:
    conn.close()