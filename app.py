"""
Databricks App: Weather + Lakebase Integration

A Flask application that:
- Connects to Lakebase (Databricks-managed Postgres)
- Fetches weather data from NWS API
- Provides interactive web UI and REST API endpoints
"""

import os
import logging
import json
from flask import Flask, request, jsonify
import lakebase
from weather_client import WeatherClient, CITY_COORDINATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Weather Client
weather_client = WeatherClient()


# ========== Database Schema Setup ==========

def ensure_tables():
    """Create tables if they don't exist."""
    try:
        # Enable pgvector extension (required for embeddings)
        try:
            lakebase.run_write("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension: {e}")
        
        # Create weather documents table
        lakebase.run_write("""
            CREATE TABLE IF NOT EXISTS weather_documents (
                id TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                source_type TEXT NOT NULL,
                headline TEXT,
                narrative_text TEXT,
                issued_at TIMESTAMPTZ,
                payload JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("weather_documents table ensured")
        
        # Create indexes
        lakebase.run_write("""
            CREATE INDEX IF NOT EXISTS idx_weather_documents_location 
            ON weather_documents (location)
        """)
        
        lakebase.run_write("""
            CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type 
            ON weather_documents (source_type)
        """)
        
        # Create weather embeddings table (384 dim for all-MiniLM-L6-v2)
        lakebase.run_write("""
            CREATE TABLE IF NOT EXISTS weather_embeddings (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                location TEXT,
                headline TEXT,
                embedding VECTOR(384) NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES weather_documents(id) ON DELETE CASCADE
            )
        """)
        logger.info("weather_embeddings table ensured")
        
        # Create HNSW index for similarity search
        try:
            lakebase.run_write("""
                CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
                ON weather_embeddings
                USING hnsw (embedding vector_cosine_ops)
            """)
            logger.info("HNSW index created")
        except Exception as e:
            logger.warning(f"Could not create HNSW index: {e}")
        
        lakebase.run_write("""
            CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id 
            ON weather_embeddings (document_id)
        """)
        
        logger.info("All database tables and indexes ensured")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


# Initialize tables on startup
try:
    ensure_tables()
except Exception as e:
    logger.warning(f"Could not initialize tables on startup: {e}")


# ========== Web UI ==========

@app.route('/')
def index():
    """Serve the interactive weather UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather + Lakebase App</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .content {
                padding: 30px;
            }
            
            .search-section {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
            }
            
            .search-section h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                font-weight: 600;
                color: #555;
                margin-bottom: 8px;
            }
            
            select, input, button {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: all 0.3s;
            }
            
            select:focus, input:focus {
                outline: none;
                border-color: #667eea;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                cursor: pointer;
                border: none;
                margin-top: 10px;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            button.danger {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            
            .filters {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .status {
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                display: none;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            
            .results {
                margin-top: 30px;
            }
            
            .results h3 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.3em;
            }
            
            .weather-card {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                transition: all 0.3s;
            }
            
            .weather-card:hover {
                border-color: #667eea;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }
            
            .weather-card .badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                margin-right: 10px;
            }
            
            .badge.alert {
                background: #f8d7da;
                color: #721c24;
            }
            
            .badge.forecast {
                background: #d1ecf1;
                color: #0c5460;
            }
            
            .weather-card h4 {
                color: #333;
                margin: 15px 0 10px 0;
                font-size: 1.2em;
            }
            
            .weather-card .meta {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 15px;
            }
            
            .weather-card .description {
                color: #555;
                line-height: 1.6;
                white-space: pre-wrap;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #999;
            }
            
            .empty-state svg {
                width: 100px;
                height: 100px;
                margin-bottom: 20px;
                opacity: 0.3;
            }
            
            .button-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌦️ Weather Dashboard</h1>
                <p>Real-time weather alerts and forecasts from the National Weather Service</p>
            </div>
            
            <div class="content">
                <!-- Sync Section -->
                <div class="search-section">
                    <h2>Step 1: Sync Weather Data</h2>
                    <p style="color: #666; margin-bottom: 15px;">Select which cities to fetch weather data for, then click Sync.</p>
                    <div class="form-group">
                        <label for="locationSelect">Select Location(s) to Sync:</label>
                        <select id="locationSelect" multiple size="7">
                            <option value="ALL">🌎 All Locations</option>
                            <option value="CHICAGO, IL">🏙️ Chicago, IL</option>
                            <option value="NEW YORK, NY">🗽 New York, NY</option>
                            <option value="LOS ANGELES, CA">🌴 Los Angeles, CA</option>
                            <option value="MIAMI, FL">🏖️ Miami, FL</option>
                            <option value="SEATTLE, WA">🌲 Seattle, WA</option>
                            <option value="DENVER, CO">⛰️ Denver, CO</option>
                            <option value="AUSTIN, TX">🤠 Austin, TX</option>
                        </select>
                        <small style="display: block; margin-top: 5px; color: #666;">Hold Ctrl/Cmd to select multiple cities</small>
                    </div>
                    <div class="button-group">
                        <button id="syncBtn" onclick="syncWeather()">🔄 Sync Selected Locations</button>
                        <button class="danger" onclick="clearAllData()">🗑️ Clear All Data</button>
                    </div>
                </div>
                
                <!-- Status Messages -->
                <div id="status" class="status"></div>
                
                <!-- Search & Filter Section -->
                <div class="search-section">
                    <h2>Step 2: View Weather Data</h2>
                    <p style="color: #666; margin-bottom: 15px;">Filter and search the weather data you've synced.</p>
                    <div class="filters">
                        <div class="form-group">
                            <label for="filterLocation">Filter by Location:</label>
                            <select id="filterLocation">
                                <option value="">🌍 Show All Locations</option>
                                <option value="CHICAGO">Chicago, IL</option>
                                <option value="NEW YORK">New York, NY</option>
                                <option value="LOS ANGELES">Los Angeles, CA</option>
                                <option value="MIAMI">Miami, FL</option>
                                <option value="SEATTLE">Seattle, WA</option>
                                <option value="DENVER">Denver, CO</option>
                                <option value="AUSTIN">Austin, TX</option>
                                <option value="IL">Illinois (All)</option>
                                <option value="NY">New York (All)</option>
                                <option value="CA">California (All)</option>
                                <option value="FL">Florida (All)</option>
                                <option value="WA">Washington (All)</option>
                                <option value="CO">Colorado (All)</option>
                                <option value="TX">Texas (All)</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="filterType">Filter by Type:</label>
                            <select id="filterType">
                                <option value="">All Types</option>
                                <option value="alert">⚠️ Alerts Only</option>
                                <option value="forecast">📊 Forecasts Only</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="filterLimit">Results Limit:</label>
                            <select id="filterLimit">
                                <option value="10">10 results</option>
                                <option value="25" selected>25 results</option>
                                <option value="50">50 results</option>
                                <option value="100">100 results</option>
                            </select>
                        </div>
                    </div>
                    <button onclick="searchWeather()">🔍 Search Weather Data</button>
                </div>
                
                <!-- Results Section -->
                <div id="results" class="results" style="display: none;">
                    <h3>📋 Weather Results <span id="resultCount"></span></h3>
                    <div id="weatherCards"></div>
                </div>
            </div>
        </div>
        
        <script>
            let lastSyncedLocations = [];
            
            // Show status message
            function showStatus(message, type = 'info') {
                const status = document.getElementById('status');
                status.className = `status ${type}`;
                status.textContent = message;
                status.style.display = 'block';
                
                if (type === 'success') {
                    setTimeout(() => {
                        status.style.display = 'none';
                    }, 5000);
                }
            }
            
            // Clear all weather data
            async function clearAllData() {
                if (!confirm('Are you sure you want to delete ALL weather data from the database?')) {
                    return;
                }
                
                try {
                    showStatus('Clearing all weather data...', 'info');
                    
                    const response = await fetch('/weather/clear', {
                        method: 'DELETE'
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showStatus(`✅ Cleared ${data.deleted_count} weather documents`, 'success');
                        document.getElementById('weatherCards').innerHTML = '';
                        document.getElementById('results').style.display = 'none';
                    } else {
                        showStatus(`❌ Error: ${data.error}`, 'error');
                    }
                } catch (error) {
                    showStatus(`❌ Network error: ${error.message}`, 'error');
                }
            }
            
            // Sync weather data
            async function syncWeather() {
                const btn = document.getElementById('syncBtn');
                const select = document.getElementById('locationSelect');
                const selectedOptions = Array.from(select.selectedOptions);
                
                btn.disabled = true;
                btn.textContent = '⏳ Syncing...';
                
                try {
                    let locations = [];
                    
                    // If "ALL" is selected or nothing selected, use all locations
                    if (selectedOptions.find(opt => opt.value === 'ALL') || selectedOptions.length === 0) {
                        locations = [];  // Empty array means fetch all
                        lastSyncedLocations = ['ALL'];
                    } else {
                        locations = selectedOptions
                            .map(opt => opt.value)
                            .filter(val => val !== 'ALL');
                        lastSyncedLocations = locations;
                    }
                    
                    showStatus(`Fetching weather data from NWS API for ${lastSyncedLocations.length === 1 && lastSyncedLocations[0] === 'ALL' ? 'all' : lastSyncedLocations.length} location(s)...`, 'info');
                    
                    const response = await fetch('/weather/sync', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ locations })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showStatus(
                            `✅ Success! Synced ${data.documents_synced} weather documents from ${data.locations_requested} locations.`,
                            'success'
                        );
                        
                        // Auto-filter to show only newly synced data
                        if (lastSyncedLocations.length === 1 && lastSyncedLocations[0] !== 'ALL') {
                            // Set filter to the synced location
                            const locationName = lastSyncedLocations[0].split(',')[0].trim().toUpperCase();
                            const filterSelect = document.getElementById('filterLocation');
                            
                            // Find matching option
                            for (let opt of filterSelect.options) {
                                if (opt.value.includes(locationName) || locationName.includes(opt.value)) {
                                    filterSelect.value = opt.value;
                                    break;
                                }
                            }
                        }
                        
                        // Auto-search after sync
                        setTimeout(() => searchWeather(), 1000);
                    } else {
                        showStatus(`❌ Error: ${data.error}`, 'error');
                    }
                } catch (error) {
                    showStatus(`❌ Network error: ${error.message}`, 'error');
                } finally {
                    btn.disabled = false;
                    btn.textContent = '🔄 Sync Selected Locations';
                }
            }
            
            // Search weather data
            async function searchWeather() {
                const locationFilter = document.getElementById('filterLocation').value;
                const type = document.getElementById('filterType').value;
                const limit = document.getElementById('filterLimit').value;
                const resultsDiv = document.getElementById('results');
                const cardsDiv = document.getElementById('weatherCards');
                
                resultsDiv.style.display = 'block';
                cardsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading weather data...</p></div>';
                
                try {
                    const params = new URLSearchParams({
                        limit: limit
                    });
                    
                    if (locationFilter) params.append('location_filter', locationFilter);
                    if (type) params.append('source_type', type);
                    
                    const response = await fetch(`/weather/documents?${params}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayWeatherData(data.documents, data.count);
                    } else {
                        cardsDiv.innerHTML = `<div class="empty-state"><p>❌ Error loading data: ${data.error}</p></div>`;
                    }
                } catch (error) {
                    cardsDiv.innerHTML = `<div class="empty-state"><p>❌ Network error: ${error.message}</p></div>`;
                }
            }
            
            // Display weather data
            function displayWeatherData(documents, count) {
                const cardsDiv = document.getElementById('weatherCards');
                const countSpan = document.getElementById('resultCount');
                
                countSpan.textContent = `(${count} found)`;
                
                if (documents.length === 0) {
                    cardsDiv.innerHTML = `
                        <div class="empty-state">
                            <svg fill="currentColor" viewBox="0 0 20 20">
                                <path d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zM10 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zM10 7a3 3 0 100 6 3 3 0 000-6zM15.657 5.404a.75.75 0 10-1.06-1.06l-1.061 1.06a.75.75 0 001.06 1.06l1.06-1.06zM6.464 14.596a.75.75 0 10-1.06-1.06l-1.06 1.06a.75.75 0 001.06 1.06l1.06-1.06zM18 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 0118 10zM5 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 015 10zM14.596 15.657a.75.75 0 001.06-1.06l-1.06-1.061a.75.75 0 10-1.06 1.06l1.06 1.06zM5.404 6.464a.75.75 0 001.06-1.06l-1.06-1.061a.75.75 0 10-1.061 1.06l1.06 1.061z" />
                            </svg>
                            <h3>No weather data found</h3>
                            <p>Try syncing weather data first or adjust your filters.</p>
                        </div>
                    `;
                    return;
                }
                
                cardsDiv.innerHTML = documents.map(doc => `
                    <div class="weather-card">
                        <div>
                            <span class="badge ${doc.source_type}">${doc.source_type === 'alert' ? '⚠️ Alert' : '📊 Forecast'}</span>
                            <span class="badge" style="background: #e7f3ff; color: #0066cc;">📍 ${doc.location}</span>
                        </div>
                        <h4>${doc.headline || 'Weather Update'}</h4>
                        <div class="meta">
                            ${doc.issued_at ? `🕐 ${new Date(doc.issued_at).toLocaleString()}` : 'Time not available'}
                        </div>
                        <div class="description">${doc.narrative_text || 'No additional details available.'}</div>
                    </div>
                `).join('');
            }
            
            // Load data on page load
            window.addEventListener('load', () => {
                searchWeather();
            });
        </script>
    </body>
    </html>
    """


# ========== API Endpoints ==========

@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        result = lakebase.run_query("SELECT 1 as health")
        return jsonify({
            "status": "healthy",
            "database": "connected" if result else "disconnected",
            "service": "weather-lakebase-app"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route('/weather/locations', methods=['GET'])
def get_weather_locations():
    """Get list of available weather locations."""
    return jsonify({
        "locations": list(CITY_COORDINATES.keys()),
        "count": len(CITY_COORDINATES),
        "note": "Default US cities. You can also provide lat,lon coordinates."
    }), 200


@app.route('/weather/sync', methods=['POST'])
def sync_weather():
    """Fetch weather data and sync to Lakebase."""
    try:
        data = request.get_json() or {}
        locations = data.get('locations', [])
        
        # If no locations provided, use default cities
        if not locations:
            locations = list(CITY_COORDINATES.keys())
        
        logger.info(f"Fetching weather data for {len(locations)} locations...")
        documents = weather_client.fetch_weather_documents(locations)
        
        # Insert into database
        inserted_count = 0
        for doc in documents:
            try:
                # Convert payload dict to JSON string
                payload_json = json.dumps(doc.get('payload')) if doc.get('payload') else None
                
                lakebase.run_write("""
                    INSERT INTO weather_documents 
                    (id, location, source_type, headline, narrative_text, issued_at, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        location = EXCLUDED.location,
                        source_type = EXCLUDED.source_type,
                        headline = EXCLUDED.headline,
                        narrative_text = EXCLUDED.narrative_text,
                        issued_at = EXCLUDED.issued_at,
                        payload = EXCLUDED.payload
                """, (
                    doc['id'],
                    doc['location'],
                    doc['source_type'],
                    doc.get('headline'),
                    doc.get('narrative_text'),
                    doc.get('issued_at'),
                    payload_json
                ))
                inserted_count += 1
            except Exception as e:
                logger.warning(f"Failed to insert document {doc['id']}: {e}")
        
        return jsonify({
            "message": "Weather sync completed",
            "locations_requested": len(locations),
            "documents_fetched": len(documents),
            "documents_synced": inserted_count
        }), 200
    except Exception as e:
        logger.error(f"Weather sync failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/weather/documents', methods=['GET'])
def get_weather_documents():
    """Get weather documents from Lakebase with improved location filtering."""
    try:
        limit = int(request.args.get('limit', 100))
        location_filter = request.args.get('location_filter', '')  # Changed parameter name
        source_type = request.args.get('source_type')
        
        query = "SELECT * FROM weather_documents WHERE 1=1"
        params = []
        
        # Improved location filtering - matches partial location strings
        if location_filter:
            query += " AND location ILIKE %s"
            params.append(f'%{location_filter}%')
        
        if source_type:
            query += " AND source_type = %s"
            params.append(source_type)
        
        query += " ORDER BY issued_at DESC LIMIT %s"
        params.append(limit)
        
        rows = lakebase.run_query(query, tuple(params))
        
        return jsonify({
            "documents": rows,
            "count": len(rows),
            "limit": limit,
            "filter_applied": location_filter if location_filter else "none"
        }), 200
    except Exception as e:
        logger.error(f"Failed to get weather documents: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/weather/clear', methods=['DELETE'])
def clear_weather_data():
    """Clear all weather documents from the database."""
    try:
        # Delete all records
        deleted = lakebase.run_write("DELETE FROM weather_documents")
        
        logger.info(f"Cleared {deleted} weather documents")
        
        return jsonify({
            "message": "All weather data cleared",
            "deleted_count": deleted
        }), 200
    except Exception as e:
        logger.error(f"Failed to clear weather data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/records', methods=['GET'])
def get_records():
    """Generic endpoint to query any table."""
    try:
        table = request.args.get('table', 'weather_documents')
        limit = int(request.args.get('limit', 100))
        
        # Whitelist allowed tables for security
        allowed_tables = ['weather_documents', 'weather_embeddings']
        if table not in allowed_tables:
            return jsonify({"error": f"Table '{table}' not allowed"}), 400
        
        rows = lakebase.run_query(f"SELECT * FROM {table} LIMIT %s", (limit,))
        
        return jsonify({
            "table": table,
            "records": rows,
            "count": len(rows)
        }), 200
    except Exception as e:
        logger.error(f"Failed to get records: {e}")
        return jsonify({"error": str(e)}), 500


# ========== Application Entry Point ==========

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Weather + Lakebase App on {host}:{port}")
    app.run(host=host, port=port, debug=False)
