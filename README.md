# Massive + Lakebase Databricks App Boilerplate

A minimal Databricks App that:
- Connects to **Lakebase** (Databricks-managed Postgres) using a single `LAKEBASE_URL` secret (a native Postgres role with a static password)
- Calls the **Massive API** (large paginated dataset) using a key stored in a Databricks secret scope
- Syncs Massive API data into Lakebase in batches
- Exposes a small Flask API to trigger syncs and read synced records

## Files

- `app.py` - Flask app: `/healthz`, `/records` (GET), `/sync` (POST)
- `lakebase.py` - Lakebase connection helper (single `LAKEBASE_URL`, psycopg2 + SQLAlchemy)
- `massive_client.py` - Massive API client with pagination generator for large datasets
- `setup_secrets.py` - One-time script to create the secret scopes and store the Massive API key + Lakebase URL
- `app.yaml` - Databricks App deployment config (command + env vars)
- `.env.example` - Local dev env var template (copy to `.env`, do not commit real values)

## Setup

1. **Create a Lakebase instance** in your Databricks workspace (Catalog > Lakebase, or via SDK/CLI).
   In the instance's **Roles & Databases** tab, add a native Postgres role with **Password**
   authentication (not OAuth) — this gives you a static, non-expiring password. Copy the
   generated connection string, e.g.:

   ```
   postgresql://<role>:<password>@<host>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
   ```

2. **Store your secrets** (run once, locally or in a notebook):

   ```bash
   python setup_secrets.py
   ```

   This prompts for your Massive API key and your Lakebase connection URL via `getpass`
   (never written to disk or shell history), and stores them as Databricks secrets:
   `massive/api-key` and `database/lakebase-url`.

3. **Configure environment variables** — copy `.env.example` to `.env` for local dev and paste
   your Lakebase URL as `LAKEBASE_URL`. For deployment, `app.yaml` already pulls `LAKEBASE_URL`
   from the `database/lakebase-url` secret automatically — no manual editing needed.

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run locally**:

   ```bash
   python app.py
   ```

6. **Deploy as a Databricks App**:

   ```bash
   databricks apps deploy <app-name> --source-code-path .
   ```

## Endpoints

- `GET /healthz` - health check
- `GET /records?limit=100` - read synced records from Lakebase
- `POST /sync?batch_size=500` with optional JSON body `{"path": "/records"}` - pull from Massive API and upsert into Lakebase

## Notes

- Lakebase auth uses a single `LAKEBASE_URL` secret pointing at a native Postgres role with a
  static, non-expiring password — no token refresh logic needed in `lakebase.py`.
- The Massive API pagination in `massive_client.py` assumes a `{"items": [...], "next_cursor": ...}`
  cursor-based shape. Adjust `paginated_get` to match the real API's pagination contract.
- For very large batch upserts, consider `psycopg2.extras.execute_values` instead of per-row inserts.
