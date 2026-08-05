# Massive + Lakebase Databricks App Boilerplate

A minimal Databricks App that:
- Connects to **Lakebase** (Databricks-managed Postgres) using a single `LAKEBASE_URL` secret (a native Postgres role with a static password)
- Calls the **Massive API** (large paginated dataset) using a key stored in a Databricks secret scope
- Syncs Massive API data into Lakebase in batches
- Exposes a small Flask API to trigger syncs and read synced records

## Files

- `app.py` - Flask app: `/healthz`, `/records` (GET), `/sync` (POST), `/watchlist` (GET/POST/DELETE), `/news/sync` (POST)
- `lakebase.py` - Lakebase connection helper (single `LAKEBASE_URL`, psycopg2 + SQLAlchemy)
- `massive_client.py` - Massive API client: pagination generator for large datasets, `get_latest_price`, `get_news`
- `setup_secrets.py` - One-time script to create the secret scopes and store the Massive API key + Lakebase URL
- `app.yaml` - Databricks App deployment config (command + env vars)
- `templates/index.html` - Watchlist UI (add + remove tickers)
- `notebooks/ingest_ticker_news_embeddings.py` - Self-contained ETL notebook: reads tickers from the `watchlist` table, fetches news for those tickers directly from Massive (rate-limited to 5 requests/min for the free API tier) into `ticker_news_documents`, computes title/description embeddings into `ticker_news_embeddings`, and fetches + chunks + embeds each article's full body (via `trafilatura`) into `ticker_news_chunk_embeddings` (pgvector)
- `databricks.yml` + `resources/ingest_ticker_news_embeddings_job.yml` - Databricks Asset Bundle config that schedules the notebook above as a Workflow (see [Scheduling the embeddings notebook](#scheduling-the-embeddings-notebook-as-a-databricks-workflow))
- `.env.example` - Local dev env var template (copy to `.env`, do not commit real values)

## Step-by-step setup

### 1. Create a Massive.com account and get an API key

1. Go to [https://massive.com](https://massive.com) and sign up for a new account (or log in if you already have one).
2. Once logged in, open your account/workspace **Settings** (or **Developer** / **API** section, depending on Massive's current UI).
3. Find **API Keys** and click **Create API Key** (or **Generate New Key**).
4. Give the key a name (e.g. `databricks-app`) and copy the generated key value immediately — most providers only show it once.
5. Keep this key handy for step 3 (Store your secrets) below. Do **not** put it in code, `.env` committed to git, or anywhere else in plaintext.

> If Massive's console differs from the steps above, look for **API Keys**, **Tokens**, or **Credentials** under your account/organization settings — the key is what authenticates requests to `https://api.massive.com` in `massive_client.py`.

### 2. Create a Lakebase instance and a native-password role

1. In your Databricks workspace, go to **Catalog** (left sidebar) and select the **Lakebase** tab (or search "Lakebase" in the workspace search bar).
2. Click **Create Lakebase instance** (sometimes labeled **Create database instance**).
   - Give it a name (e.g. `massive-sync-db`).
   - Choose the capacity/compute size and region appropriate for your workload (defaults are fine to start).
   - Click **Create** and wait for the instance to reach the **Available**/**Running** state.
3. Open the newly created instance, then go to the **Roles & Databases** tab (sometimes called **Permissions** or **Roles**).
4. **Enable native (password) authentication** for the instance if it isn't already on:
   - Look for an authentication setting such as **Native passwords** or **Password authentication** and toggle/enable it. By default some Lakebase instances only support OAuth/token-based auth — you need password auth enabled so the role below gets a static password instead of a short-lived token.
5. **Create a new role**:
   - Click **Add role** / **Create role**.
   - Choose **Password** as the authentication method (not OAuth).
   - Name the role (e.g. `massive_app`) and let Databricks generate (or set) a password.
6. **Copy the connection URL** shown for the role. It will look like:

   ```
   postgresql://<role>:<password>@<host>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
   ```

   Keep this URL — you'll paste it into `setup_secrets.py`'s prompt in the next step.

### 3. Store your secrets

Run once from a **Databricks notebook** in your workspace (no CLI needed):

1. Create a new notebook (or open the Git folder you'll create in step 5, once it's cloned) and attach it to any running cluster.
2. In a cell, run:

   ```python
   %sh python setup_secrets.py
   ```

   or open a terminal from the notebook (**Run** > **Open terminal**, if enabled on your cluster) and run `python setup_secrets.py` there.

This prompts (via `getpass`, so nothing is echoed or written to disk/shell history) for:
- Your **Massive API key** (from step 1) → stored as secret `massive/api-key`
- Your **Lakebase connection URL** (from step 2) → stored as secret `database/lakebase-url`

### 4. Configure environment variables (local dev)

Copy `.env.example` to `.env` and paste your Lakebase URL as `LAKEBASE_URL` for local runs:

```bash
cp .env.example .env
```

For deployment, `app.yaml` already pulls `LAKEBASE_URL` from the `database/lakebase-url` secret automatically — no manual editing needed there.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run locally

```bash
python app.py
```

### 7. Create a Git folder in Databricks and deploy the app (no CLI required)

All of this is done through the Databricks workspace UI:

1. **Create a Git folder**:
   - In the Databricks workspace sidebar, click **Workspace** > **Create** > **Git folder** (in older UIs this is called **Repos** > **Add Repo**).
   - Paste the Git URL of this project's repository (e.g. your GitHub/GitLab remote for this codebase).
   - Choose a folder name and click **Create Git folder**. Databricks will clone the repo directly into your workspace — this becomes the source for your app.

2. **Create the Databricks App**:
   - In the sidebar, go to **Compute** > **Apps** (or search "Apps" in the workspace search bar).
   - Click **Create app**, then choose **Custom** (or "From scratch").
   - Give the app a name (e.g. `massive-lakebase-sync`).

3. **Point the app at your Git folder**:
   - When prompted for the source code location, select **Workspace files** / **Git folder** and browse to the Git folder you created in step 1 (the folder containing `app.py` and `app.yaml`).
   - Databricks will read `app.yaml` from that folder automatically to configure the `command` and `env` (including the `LAKEBASE_URL`, `MASSIVE_API_BASE_URL`, and secret scope/key references).

4. **Deploy**:
   - Click **Deploy** (or **Create and deploy**) in the Apps UI. Databricks will build and start the app using the Git folder's current contents — no `databricks` CLI commands are needed.
   - Whenever you update the code, pull the latest changes into the Git folder (**Git folder** > **Pull**, via the UI) and click **Deploy** again in the Apps UI to redeploy.

5. Once deployed, open the app's URL from the Apps UI and hit `GET /healthz` to confirm it's running, then try `POST /sync` to pull data from Massive into Lakebase.

## Endpoints

- `GET /healthz` - health check
- `GET /records?limit=100` - read synced records from Lakebase
- `POST /sync?batch_size=500` with optional JSON body `{"path": "/records"}` - pull from Massive API and upsert into Lakebase
- `GET /watchlist` - get the current user's watchlist symbols with last known price
- `POST /watchlist` - add/update a symbol on the current user's watchlist
- `DELETE /watchlist/<symbol>` - remove a symbol from the current user's watchlist
- `POST /news/sync` with optional JSON body `{"tickers": ["AAPL", "MSFT"], "limit": 50}` - pull recent news per ticker from Massive and upsert into `ticker_news_documents`

## Scheduling the embeddings notebook as a Databricks Workflow

`notebooks/ingest_ticker_news_embeddings.py` is a self-contained ETL: it reads the distinct
tickers from the `watchlist` table, fetches news for those tickers directly from Massive
(serially, rate-limited to `max_requests_per_minute` - 5/min by default, matching the free
Massive API tier's strict limits), and upserts them into `ticker_news_documents`. It then turns
those rows into vector embeddings in `ticker_news_embeddings` (title + description) and
`ticker_news_chunk_embeddings` (chunks of the full article body, fetched from each article's
`article_url` and extracted with `trafilatura`). You can run it on a schedule two ways — pick
whichever fits your setup:

### Option A: Databricks Asset Bundle (CLI, version-controlled)

This repo already includes bundle config for this: `databricks.yml` +
`resources/ingest_ticker_news_embeddings_job.yml`. This is the recommended path if you want the
job definition tracked in git alongside the code.

1. Set the real workspace URL in `databricks.yml` (replace `<your-workspace-instance>`).
2. Deploy: `databricks bundle deploy -t dev`
3. Test it once manually: `databricks bundle run ingest_ticker_news_embeddings_job -t dev`
4. Once you've confirmed a successful run, flip `pause_status: PAUSED` to `pause_status: UNPAUSED`
   in `resources/ingest_ticker_news_embeddings_job.yml` and redeploy to turn on the daily schedule.

### Option B: Workflows UI (no CLI required)

If you'd rather not use the CLI, you can create the equivalent job by hand in the Databricks UI:

1. **Get the notebook into your workspace**: if you already created a Git folder for this repo
   (see step 7 above), the notebook is already there at `notebooks/ingest_ticker_news_embeddings.py`.
   Otherwise, upload/import it via **Workspace** > **Create** > **Notebook** > **Import**.
2. **Create the job**: go to **Workflows** (left sidebar) > **Jobs** > **Create Job**.
3. **Add a task**:
   - Task type: **Notebook**.
   - Notebook path: browse to `notebooks/ingest_ticker_news_embeddings.py` in your Git folder.
   - Cluster: choose **New job cluster** (a small general-purpose cluster is enough) or an existing
     cluster/serverless, if available.
   - Under **Parameters**, add the same widget values the notebook expects:
     - `watchlist_table_name` = `watchlist`
     - `news_table_name` = `ticker_news_documents`
     - `embeddings_table_name` = `ticker_news_embeddings`
     - `chunk_embeddings_table_name` = `ticker_news_chunk_embeddings`
     - `embedding_model` = `sentence-transformers/all-MiniLM-L6-v2`
     - `massive_secret_scope` = `massive`
     - `massive_secret_key` = `api-key`
     - `massive_api_base_url` = `https://api.massive.com`
     - `news_fetch_limit` = `50`
     - `max_requests_per_minute` = `5`
     - `chunk_size` = `800`
     - `chunk_overlap` = `100`
4. **Add a schedule**: click **Add trigger** on the job, choose **Scheduled**, and set it to run
   daily (e.g. 6:00 AM UTC) using either the simple picker or a cron expression
   (`0 0 6 * * ?`, timezone UTC).
5. **Add a failure notification**: under **Notifications**, add your email/Slack webhook for
   on-failure alerts.
6. Click **Create** and optionally **Run now** to validate the job before its first scheduled run.

Both options produce the same result — a Databricks Workflow that runs the notebook and refreshes
`ticker_news_embeddings`. The Asset Bundle keeps the definition in git and reproducible across
workspaces; the UI path is quicker for a one-off class demo but isn't tracked in version control.

## Enabling Change Data Feed (CDF) for Postgres tables

Lakebase supports **Change Data Feed (CDF)**, a managed way to stream row-level inserts/updates/deletes
from your Lakebase Postgres tables into Unity Catalog Delta tables (no Debezium, no custom connectors).
CDF is enabled per-**schema** in the `databricks_postgres` database, and every table in that schema that
meets two conditions is picked up automatically: it has `REPLICA IDENTITY FULL` set, and it has at least
one row.

> **Note:** CDF is only available on paid Databricks accounts — it is not supported on the free
> Databricks Community Edition or trial tier.

### 1. Set `REPLICA IDENTITY FULL` on the tables you want to track

By default, Postgres only logs primary-key columns on change. To capture full row contents (needed for
CDF), enable `REPLICA IDENTITY FULL` on each table — including `watchlist` and `massive_records` from
this app:

```sql
ALTER TABLE watchlist REPLICA IDENTITY FULL;
ALTER TABLE massive_records REPLICA IDENTITY FULL;
```

Run this once per table, either from a Databricks SQL editor connected to your Lakebase instance, or
from a `psql` session using your `LAKEBASE_URL`. Any new table you add later (e.g. via `ensure_table`-style
helpers in `app.py`) needs the same `ALTER TABLE ... REPLICA IDENTITY FULL` statement run once before it
will be included in the feed. Tables with the setting but zero rows are skipped until the first row is
inserted, then picked up automatically.

You can confirm which tables currently qualify by querying:

```sql
SELECT * FROM wal2delta.tables;
```

### 2. Start CDF from the Lakebase UI

1. In your Databricks workspace, open the **Lakebase** tab for your instance.
2. Go to **Lakebase CDF** and click **Start**.
3. Select the `databricks_postgres` database and the schema containing your tables (the default
   schema, `public`, works — it's inside `databricks_postgres`).
4. Choose the Unity Catalog destination schema/catalog where the CDF history tables should land.
5. Confirm — the UI shows a preview of qualifying tables (e.g. `watchlist`, `massive_records`) and
   their sync status before you start.

Once running, each qualifying table gets a corresponding Delta table named `lb_<table_name>_history`
(e.g. `lb_watchlist_history`) in Unity Catalog, updated roughly every 15 seconds. Each row includes
metadata columns (`_pg_change_type`, `_pg_lsn`, `_pg_xid`, `_timestamp`, `_sort_by`) describing the
change, so downstream Delta Live Tables/pipelines can build Silver/Gold layers off the append-only
history.

> **Note:** Disabling CDF is lossy — changes made while it's off aren't captured, and re-enabling
> triggers a full resync (every row reloaded as an `insert`). There's no per-table exclusion option
> within an enabled schema; the only way to keep a table out of the feed is to not set
> `REPLICA IDENTITY FULL` on it.

## Notes

- Lakebase auth uses a single `LAKEBASE_URL` secret pointing at a native Postgres role with a
  static, non-expiring password — no token refresh logic needed in `lakebase.py`.
- The Massive API pagination in `massive_client.py` assumes a `{"items": [...], "next_cursor": ...}`
  cursor-based shape. Adjust `paginated_get` to match the real API's pagination contract.
- For very large batch upserts, consider `psycopg2.extras.execute_values` instead of per-row inserts.
