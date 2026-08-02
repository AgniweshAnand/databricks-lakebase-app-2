# Massive + Lakebase Databricks App Boilerplate

A minimal Databricks App that:
- Connects to **Lakebase** (Databricks-managed Postgres) using short-lived OAuth tokens
- Calls the **Massive API** (large paginated dataset) using a key stored in a Databricks secret scope
- Syncs Massive API data into Lakebase in batches
- Exposes a small Flask API to trigger syncs and read synced records

## Files

- `app.py` - Flask app: `/healthz`, `/records` (GET), `/sync` (POST)
- `lakebase.py` - Lakebase connection helper (OAuth token via Databricks SDK, psycopg2 + SQLAlchemy)
- `massive_client.py` - Massive API client with pagination generator for large datasets
- `setup_secrets.py` - One-time script to create the secret scope and store the Massive API key
- `app.yaml` - Databricks App deployment config (command + env vars)
- `.env.example` - Local dev env var template (copy to `.env`, do not commit real values)

## Setup

1. **Store your Massive API key as a Databricks secret** (run once, locally or in a notebook):

   ```bash
   python setup_secrets.py
   ```

   This creates scope `massive` and secret key `api-key` and prompts for the value via `getpass`
   so it's never written to disk or shell history.

2. **Create a Lakebase instance** in your Databricks workspace (Catalog > Lakebase, or via SDK/CLI)
   and note its instance name + host.

3. **Configure environment variables** — copy `.env.example` to `.env` for local dev, or edit
   `app.yaml` for deployment:
   - `LAKEBASE_INSTANCE_NAME`, `LAKEBASE_HOST`, `LAKEBASE_PORT`, `LAKEBASE_DATABASE`, `LAKEBASE_USER`
   - `MASSIVE_API_BASE_URL`, `MASSIVE_SECRET_SCOPE`, `MASSIVE_SECRET_KEY`

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

- Lakebase auth uses short-lived OAuth tokens (via `WorkspaceClient().database.generate_database_credential`),
  not static passwords — `lakebase.py` handles refreshing them.
- The Massive API pagination in `massive_client.py` assumes a `{"items": [...], "next_cursor": ...}`
  cursor-based shape. Adjust `paginated_get` to match the real API's pagination contract.
- For very large batch upserts, consider `psycopg2.extras.execute_values` instead of per-row inserts.
