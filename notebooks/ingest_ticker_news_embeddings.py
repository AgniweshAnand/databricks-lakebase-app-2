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

# MAGIC %pip install -q sentence-transformers trafilatura requests

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
# MAGIC stored in a Databricks secret scope. We parse it into the pieces both
# MAGIC Spark's JDBC reader AND the raw JDBC connection helper below need
# MAGIC (url/user/password).

# COMMAND ----------

import base64
from urllib.parse import urlparse, quote_plus

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Build JDBC URL without credentials
jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
print(f"Connecting to: {parsed.hostname}:{parsed.port or 5432}{parsed.path}")

# Pass credentials and SSL settings in properties
jdbc_properties = {
    "user": parsed.username,
    "password": parsed.password,
    "driver": "org.postgresql.Driver",
    "sslmode": "require",
}

print(f"User: {parsed.username}")
print(f"Database: {parsed.path}")

# COMMAND ----------

# DBTITLE 1,Test JDBC Connection
# Test JDBC connection with embedded credentials
try:
    test_df = spark.read.jdbc(
        url=jdbc_url,
        table=WATCHLIST_TABLE_NAME,
        properties=jdbc_properties
    )
    count = test_df.count()
    print(f"✅ Connection successful! Found {count} rows in {WATCHLIST_TABLE_NAME}")
    test_df.show(5)
except Exception as e:
    print(f"❌ Connection failed: {e}")

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
# MAGIC This notebook uses Spark JDBC for all database operations - no psycopg2 required.

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

import base64 as _b64
import json as _json
import time
from datetime import datetime

import requests
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import StringType, StructField, StructType


def get_massive_api_key() -> str:
    secret = w.secrets.get_secret(scope=MASSIVE_SECRET_SCOPE, key=MASSIVE_SECRET_KEY)
    return _b64.b64decode(secret.value).decode("utf-8")


def get_watchlist_tickers() -> list[str]:
    """Distinct, uppercased ticker symbols currently tracked across all users
    in the watchlist table - these are the only tickers we fetch news for."""
    watchlist_df = spark.read.jdbc(
        url=jdbc_url, table=WATCHLIST_TABLE_NAME, properties=jdbc_properties
    )
    symbols = watchlist_df.select("symbol").distinct().collect()
    return [row.symbol.strip().upper() for row in symbols if row.symbol]


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


def sync_news_to_spark(ticker: str, articles: list[dict]):
    """Convert Massive API response to Spark DataFrame and write to Postgres.
    Uses append mode with deduplication by reading existing IDs first."""
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
        rows.append(
            {
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
            }
        )

    if not rows:
        return 0

    # Create DataFrame from API response
    new_df = spark.createDataFrame(rows).withColumn("synced_at", current_timestamp())

    # Read existing article IDs to avoid duplicates
    try:
        existing_ids = (
            spark.read.jdbc(url=jdbc_url, table=NEWS_TABLE_NAME, properties=jdbc_properties)
            .select("id")
            .distinct()
        )
        # Filter out existing IDs
        new_df = new_df.join(existing_ids, on="id", how="left_anti")
    except Exception:
        # Table might be empty or not exist yet - that's fine, write all rows
        pass

    # Write new records
    if new_df.count() > 0:
        new_df.write.jdbc(
            url=jdbc_url, table=NEWS_TABLE_NAME, mode="append", properties=jdbc_properties
        )
        return new_df.count()
    return 0


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
for i, ticker in enumerate(tickers):
    if i > 0:
        time.sleep(_seconds_between_requests)
    try:
        articles = fetch_news_for_ticker(_massive_session, ticker, NEWS_FETCH_LIMIT)
        news_synced += sync_news_to_spark(ticker, articles)
    except Exception as exc:
        print(f"Skipping {ticker}: failed to fetch/sync news ({exc})")
        continue

print(f"Synced {news_synced} new news articles into {NEWS_TABLE_NAME} for {len(tickers)} tickers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw news documents with Spark
# MAGIC
# MAGIC Reads the whole `ticker_news_documents` table (just synced from Massive
# MAGIC above) via JDBC into a Spark DataFrame so embedding computation can be
# MAGIC distributed across the cluster.

# COMMAND ----------

news_df = (
    spark.read.jdbc(url=jdbc_url, table=NEWS_TABLE_NAME, properties=jdbc_properties)
    .selectExpr(
        "id",
        "ticker",
        "title",
        "description",
        "article_url",
        "published_utc",
        # Embed on title + description together for richer context.
        "trim(concat(coalesce(title, ''), '. ', coalesce(description, ''))) AS embedding_text",
    )
    .filter("embedding_text IS NOT NULL AND embedding_text != ''")
)

print(f"Loaded {news_df.count()} news documents from {NEWS_TABLE_NAME}")
display(news_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings (distributed pandas UDF)
# MAGIC
# MAGIC Loads the sentence-transformers model once per executor process (not per
# MAGIC row) and applies it in batches via `mapInPandas`, which scales across
# MAGIC however many workers the cluster has.

# COMMAND ----------

from typing import Iterator

import pandas as pd
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

embeddings_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("ticker", StringType(), False),
        StructField("title", StringType(), False),
        StructField("published_utc", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), False),
    ]
)


def embed_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition/task: load the model once, then embed
    every batch of rows handed to this partition."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for batch in iterator:
        vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
        yield pd.DataFrame(
            {
                "id": batch["id"],
                "ticker": batch["ticker"],
                "title": batch["title"],
                "published_utc": batch["published_utc"].astype(str),
                "embedding": [v.tolist() for v in vectors],
            }
        )


embeddings_df = news_df.mapInPandas(embed_partitions, schema=embeddings_schema)

print(f"Computed {embeddings_df.count()} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC `pgvector` isn't a JDBC-native type, but plain SQL text (`vector(N)`,
# MAGIC `::vector` casts) works fine over a raw JDBC connection - no psycopg2
# MAGIC needed.

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
# MAGIC Written in batches via JDBC's `addBatch`/`executeBatch` for throughput.
# MAGIC Each embedding is cast to Postgres' `vector` type via `::vector`.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, expr, lit
from pyspark.sql.types import DoubleType

# Add model_name and embedded_at columns
embeddings_with_meta = embeddings_df.withColumn("model_name", lit(EMBEDDING_MODEL_NAME)).withColumn(
    "embedded_at", current_timestamp()
)

# Convert embedding array from ArrayType(FloatType) to ArrayType(DoubleType)
# Postgres JDBC expects DOUBLE PRECISION[] for vector columns
embeddings_final = embeddings_with_meta.withColumn(
    "embedding", expr("transform(embedding, x -> cast(x as double))")
)

# Read existing embedding IDs to avoid duplicates (deduplication strategy)
try:
    existing_ids = (
        spark.read.jdbc(url=jdbc_url, table=EMBEDDINGS_TABLE_NAME, properties=jdbc_properties)
        .select("id")
        .distinct()
    )
    # Filter out existing IDs (left anti-join keeps only new records)
    new_embeddings = embeddings_final.join(existing_ids, on="id", how="left_anti")
except Exception:
    # Table might be empty or not exist - write all embeddings
    new_embeddings = embeddings_final

embedding_count = new_embeddings.count()
if embedding_count > 0:
    # NOTE: Spark JDBC does not natively support pgvector VECTOR type.
    # The embeddings are written as DOUBLE PRECISION[] arrays.
    # You'll need to manually cast them to VECTOR in a post-processing SQL step:
    #   UPDATE ticker_news_embeddings 
    #   SET embedding = embedding::vector 
    #   WHERE embedding IS NOT NULL;
    new_embeddings.write.jdbc(
        url=jdbc_url, table=EMBEDDINGS_TABLE_NAME, mode="append", properties=jdbc_properties
    )
    print(f"Wrote {embedding_count} new embeddings to {EMBEDDINGS_TABLE_NAME}")
    print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
    print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector;")
else:
    print("No new embeddings to write (all already exist).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch and chunk article content
# MAGIC
# MAGIC Title/description only gets you so far - the actual article body lives at
# MAGIC `article_url` on the publisher's site. This step fetches each URL, uses
# MAGIC `trafilatura` to extract just the article text (stripping nav/ads/related
# MAGIC links/etc.), and splits it into overlapping chunks so each chunk can be
# MAGIC embedded and retrieved independently. Fetching is distributed across the
# MAGIC cluster via `mapInPandas`; any URL that fails to fetch/extract (paywall,
# MAGIC timeout, dead link) is skipped rather than failing the whole job.

# COMMAND ----------

content_df = news_df.select("id", "ticker", "article_url").filter(
    "article_url IS NOT NULL AND article_url != ''"
)

chunks_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("ticker", StringType(), False),
        StructField("chunk_index", StringType(), False),
        StructField("chunk_text", StringType(), False),
    ]
)


def fetch_and_chunk_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition/task: fetch each article's HTML, extract
    the main body text with trafilatura, then split it into overlapping
    chunks of CHUNK_SIZE characters (CHUNK_OVERLAP characters shared between
    consecutive chunks so context isn't lost at chunk boundaries)."""
    import requests
    import trafilatura

    for batch in iterator:
        out_article_ids, out_tickers, out_chunk_indexes, out_chunk_texts = [], [], [], []
        for article_id, ticker, article_url in zip(
            batch["id"], batch["ticker"], batch["article_url"]
        ):
            try:
                resp = requests.get(article_url, timeout=15)
                resp.raise_for_status()
                text = trafilatura.extract(resp.text)
            except Exception:
                # Dead link, paywall, timeout, etc. - skip this article's
                # content chunks rather than failing the whole job.
                continue

            if not text:
                continue

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

        yield pd.DataFrame(
            {
                "article_id": out_article_ids,
                "ticker": out_tickers,
                "chunk_index": out_chunk_indexes,
                "chunk_text": out_chunk_texts,
            }
        )


chunks_df = content_df.mapInPandas(fetch_and_chunk_partitions, schema=chunks_schema)

print(f"Extracted {chunks_df.count()} content chunks from {content_df.count()} article URLs")
display(chunks_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings
# MAGIC
# MAGIC Same approach as the title/description embeddings above, but one vector
# MAGIC per content chunk instead of per article.

# COMMAND ----------

chunk_embeddings_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("ticker", StringType(), False),
        StructField("chunk_index", StringType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
    ]
)


def embed_chunk_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition: load the model once, then embed
    every batch of chunks handed to this partition."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for batch in iterator:
        vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
        yield pd.DataFrame(
            {
                "article_id": batch["article_id"],
                "ticker": batch["ticker"],
                "chunk_index": batch["chunk_index"],
                "chunk_text": batch["chunk_text"],
                "embedding": [v.tolist() for v in vectors],
            }
        )


chunk_embeddings_df = chunks_df.mapInPandas(embed_chunk_partitions, schema=chunk_embeddings_schema)

print(f"Computed {chunk_embeddings_df.count()} chunk embeddings using {EMBEDDING_MODEL_NAME}")

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

# Add id (article_id_chunk_index), model_name, and embedded_at columns
chunk_embeddings_with_meta = (
    chunk_embeddings_df.withColumn(
        "id", expr("concat(article_id, '_', chunk_index)")
    )
    .withColumn("model_name", lit(EMBEDDING_MODEL_NAME))
    .withColumn("embedded_at", current_timestamp())
    .withColumn("chunk_index", col("chunk_index").cast("int"))
)

# Convert embedding array from ArrayType(FloatType) to ArrayType(DoubleType)
# Postgres JDBC expects DOUBLE PRECISION[] for vector columns
chunk_embeddings_final = chunk_embeddings_with_meta.withColumn(
    "embedding", expr("transform(embedding, x -> cast(x as double))")
)

# Read existing chunk embedding IDs to avoid duplicates (deduplication strategy)
try:
    existing_ids = (
        spark.read.jdbc(url=jdbc_url, table=CHUNK_EMBEDDINGS_TABLE_NAME, properties=jdbc_properties)
        .select("id")
        .distinct()
    )
    # Filter out existing IDs (left anti-join keeps only new records)
    new_chunk_embeddings = chunk_embeddings_final.join(existing_ids, on="id", how="left_anti")
except Exception:
    # Table might be empty or not exist - write all chunk embeddings
    new_chunk_embeddings = chunk_embeddings_final

chunk_count = new_chunk_embeddings.count()
if chunk_count > 0:
    # NOTE: Spark JDBC does not natively support pgvector VECTOR type.
    # The embeddings are written as DOUBLE PRECISION[] arrays.
    # You'll need to manually cast them to VECTOR in a post-processing SQL step:
    #   UPDATE ticker_news_chunk_embeddings 
    #   SET embedding = embedding::vector 
    #   WHERE embedding IS NOT NULL;
    new_chunk_embeddings.write.jdbc(
        url=jdbc_url, table=CHUNK_EMBEDDINGS_TABLE_NAME, mode="append", properties=jdbc_properties
    )
    print(f"Wrote {chunk_count} new chunk embeddings to {CHUNK_EMBEDDINGS_TABLE_NAME}")
    print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
    print(f"  UPDATE {CHUNK_EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector;")
else:
    print("No new chunk embeddings to write (all already exist).")
