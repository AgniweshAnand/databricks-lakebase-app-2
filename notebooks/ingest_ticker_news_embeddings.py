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
# MAGIC It re-uses the SAME Lakebase secret (`LAKEBASE_SECRET_SCOPE` /
# MAGIC `LAKEBASE_SECRET_KEY`) that `lakebase.py` uses in the Flask app, so no
# MAGIC extra secrets need to be created for this notebook.

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers pgvector trafilatura requests

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
dbutils.widgets.text("lakebase_secret_scope", "database", "Lakebase secret scope")
dbutils.widgets.text("lakebase_secret_key", "lakebase-url", "Lakebase secret key")
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
LAKEBASE_SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
MASSIVE_SECRET_SCOPE = dbutils.widgets.get("massive_secret_scope")
MASSIVE_SECRET_KEY = dbutils.widgets.get("massive_secret_key")
MASSIVE_API_BASE_URL = dbutils.widgets.get("massive_api_base_url")
NEWS_FETCH_LIMIT = int(dbutils.widgets.get("news_fetch_limit"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))
LAKEBASE_SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")
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
# MAGIC stored in a Databricks secret scope. We parse it into the pieces Spark's
# MAGIC JDBC reader needs (url/user/password) and also keep the raw DSN for the
# MAGIC psycopg2 writer step (pgvector values can't go through the JDBC writer).

# COMMAND ----------

import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope=LAKEBASE_SECRET_SCOPE, key=LAKEBASE_SECRET_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
jdbc_properties = {
    "user": parsed.username,
    "password": parsed.password,
    "driver": "org.postgresql.Driver",
    "sslmode": "require",
}

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
import time

import psycopg2
import requests


def get_massive_api_key() -> str:
    secret = w.secrets.get_secret(scope=MASSIVE_SECRET_SCOPE, key=MASSIVE_SECRET_KEY)
    return _b64.b64decode(secret.value).decode("utf-8")


def get_watchlist_tickers() -> list[str]:
    """Distinct, uppercased ticker symbols currently tracked across all users
    in the watchlist table - these are the only tickers we fetch news for."""
    with psycopg2.connect(lakebase_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT symbol FROM {WATCHLIST_TABLE_NAME}")
            return [row[0].strip().upper() for row in cur.fetchall() if row[0]]


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


def ensure_news_table():
    """Same schema as ensure_news_table() in app.py, kept in sync so either
    the Flask app's POST /news/sync or this notebook can populate the table."""
    with psycopg2.connect(lakebase_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {NEWS_TABLE_NAME} (
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
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{NEWS_TABLE_NAME}_ticker "
                f"ON {NEWS_TABLE_NAME} (ticker)"
            )
        conn.commit()


def upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    """Same upsert logic as _upsert_news_batch() in app.py."""
    import json as _json

    count = 0
    with psycopg2.connect(lakebase_url) as conn:
        with conn.cursor() as cur:
            for article in articles:
                sentiment = None
                sentiment_reasoning = None
                for insight in article.get("insights", []) or []:
                    if insight.get("ticker") == ticker:
                        sentiment = insight.get("sentiment")
                        sentiment_reasoning = insight.get("sentiment_reasoning")
                        break

                publisher = article.get("publisher") or {}
                cur.execute(
                    f"""
                    INSERT INTO {NEWS_TABLE_NAME} (
                        id, ticker, title, description, author, article_url,
                        publisher_name, keywords, sentiment, sentiment_reasoning,
                        published_utc, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            article_url = EXCLUDED.article_url,
                            publisher_name = EXCLUDED.publisher_name,
                            keywords = EXCLUDED.keywords,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                            published_utc = EXCLUDED.published_utc,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        str(article.get("id")),
                        ticker,
                        article.get("title", ""),
                        article.get("description"),
                        article.get("author"),
                        article.get("article_url"),
                        publisher.get("name"),
                        _json.dumps(article.get("keywords", [])),
                        sentiment,
                        sentiment_reasoning,
                        article.get("published_utc"),
                        _json.dumps(article),
                    ),
                )
                count += 1
            conn.commit()
    return count


ensure_news_table()

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
    except Exception as exc:
        print(f"Skipping {ticker}: failed to fetch news ({exc})")
        continue
    news_synced += upsert_news_batch(ticker, articles)

print(f"Synced {news_synced} news articles into {NEWS_TABLE_NAME} for {len(tickers)} tickers")

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
embeddings_pdf = embeddings_df.toPandas()

print(f"Computed {len(embeddings_pdf)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC `pgvector` isn't a JDBC-native type, so we create the table (and enable
# MAGIC the extension) with a direct psycopg2 connection rather than Spark's JDBC
# MAGIC writer, which can't express a `vector(N)` column.

# COMMAND ----------

import psycopg2

with psycopg2.connect(lakebase_url) as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                title TEXT NOT NULL,
                published_utc TIMESTAMPTZ,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                model_name TEXT NOT NULL,
                embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Approximate-nearest-neighbor index for fast cosine similarity search.
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_embedding
            ON {EMBEDDINGS_TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written in batches with `psycopg2.extras.execute_values` for throughput.
# MAGIC Each embedding is cast to Postgres' `vector` type via `::vector`.

# COMMAND ----------

from psycopg2.extras import execute_values


def _to_pgvector_literal(vector: list[float]) -> str:
    """pgvector accepts vectors as a string literal like '[0.1,0.2,...]'."""
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


rows = [
    (
        row["id"],
        row["ticker"],
        row["title"],
        row["published_utc"],
        _to_pgvector_literal(row["embedding"]),
        EMBEDDING_MODEL_NAME,
    )
    for _, row in embeddings_pdf.iterrows()
]

BATCH_SIZE = 500
upserted = 0

with psycopg2.connect(lakebase_url) as conn:
    with conn.cursor() as cur:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            execute_values(
                cur,
                f"""
                INSERT INTO {EMBEDDINGS_TABLE_NAME}
                    (id, ticker, title, published_utc, embedding, model_name, embedded_at)
                VALUES %s
                ON CONFLICT (id) DO UPDATE
                    SET ticker = EXCLUDED.ticker,
                        title = EXCLUDED.title,
                        published_utc = EXCLUDED.published_utc,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        embedded_at = EXCLUDED.embedded_at
                """,
                batch,
                template="(%s, %s, %s, %s, %s::vector, %s, now())",
            )
            upserted += len(batch)
        conn.commit()

print(f"Upserted {upserted} embeddings into {EMBEDDINGS_TABLE_NAME}")

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
chunks_pdf = chunks_df.toPandas()

print(f"Extracted {len(chunks_pdf)} content chunks from {content_df.count()} article URLs")
display(chunks_pdf.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings
# MAGIC
# MAGIC Same approach as the title/description embeddings above, but one vector
# MAGIC per content chunk instead of per article.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

_chunk_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chunks_pdf["embedding"] = (
    list(_chunk_model.encode(chunks_pdf["chunk_text"].tolist(), show_progress_bar=False))
    if len(chunks_pdf)
    else []
)

print(f"Computed {len(chunks_pdf)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the chunk embeddings destination table exists

# COMMAND ----------

with psycopg2.connect(lakebase_url) as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CHUNK_EMBEDDINGS_TABLE_NAME} (
                id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                chunk_index INT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                model_name TEXT NOT NULL,
                embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Approximate-nearest-neighbor index for fast cosine similarity search.
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{CHUNK_EMBEDDINGS_TABLE_NAME}_embedding
            ON {CHUNK_EMBEDDINGS_TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase

# COMMAND ----------

chunk_rows = [
    (
        f"{row['article_id']}_{row['chunk_index']}",
        row["article_id"],
        row["ticker"],
        int(row["chunk_index"]),
        row["chunk_text"],
        _to_pgvector_literal(row["embedding"]),
        EMBEDDING_MODEL_NAME,
    )
    for _, row in chunks_pdf.iterrows()
]

chunk_upserted = 0

with psycopg2.connect(lakebase_url) as conn:
    with conn.cursor() as cur:
        for start in range(0, len(chunk_rows), BATCH_SIZE):
            batch = chunk_rows[start : start + BATCH_SIZE]
            execute_values(
                cur,
                f"""
                INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME}
                    (id, article_id, ticker, chunk_index, chunk_text, embedding, model_name, embedded_at)
                VALUES %s
                ON CONFLICT (id) DO UPDATE
                    SET article_id = EXCLUDED.article_id,
                        ticker = EXCLUDED.ticker,
                        chunk_index = EXCLUDED.chunk_index,
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        embedded_at = EXCLUDED.embedded_at
                """,
                batch,
                template="(%s, %s, %s, %s, %s, %s::vector, %s, now())",
            )
            chunk_upserted += len(batch)
        conn.commit()

print(f"Upserted {chunk_upserted} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}")

