"""
Client for the National Weather Service (NWS) API (api.weather.gov).

No API key required, but NWS requires a custom User-Agent header.
Fetches active weather alerts and point forecasts, normalizing them
into a common document schema.
"""

import hashlib
import logging
from typing import Any
import requests

logger = logging.getLogger("weather-client")

NWS_BASE_URL = "https://api.weather.gov"
USER_AGENT = "(DatabricksWeatherApp/1.0, contact@example.com)"

# Coordinate lookup table for common US cities
CITY_COORDINATES = {
    "CHICAGO, IL": (41.8781, -87.6298),
    "AUSTIN, TX": (30.2672, -97.7431),
    "NEW YORK, NY": (40.7128, -74.0060),
    "LOS ANGELES, CA": (34.0522, -118.2437),
    "MIAMI, FL": (25.7617, -80.1918),
    "SEATTLE, WA": (47.6062, -122.3321),
    "DENVER, CO": (39.7392, -104.9903),
}


class WeatherClient:
    """HTTP Client for api.weather.gov"""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_coordinates(self, location_str: str) -> tuple[float, float] | None:
        """Resolve a city/state string or raw 'lat,lon' string to (lat, lon)."""
        loc_clean = location_str.strip().upper()
        if loc_clean in CITY_COORDINATES:
            return CITY_COORDINATES[loc_clean]

        # Check if formatted as "lat, lon"
        if "," in location_str:
            parts = location_str.split(",")
            if len(parts) == 2:
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    pass
        return None

    def get_alerts_by_area(self, state: str) -> list[dict]:
        """Fetch active alerts for a US state (2-letter code, e.g. 'IL', 'TX')."""
        url = f"{NWS_BASE_URL}/alerts/active/area/{state.upper()}"
        data = self._get(url)
        features = data.get("features", [])

        documents = []
        for feat in features:
            props = feat.get("properties", {})
            alert_id = props.get("id") or feat.get("id")

            desc = props.get("description", "") or ""
            inst = props.get("instruction", "") or ""
            narrative = f"{desc}\n\nInstructions:\n{inst}".strip()

            if not narrative:
                continue

            documents.append(
                {
                    "id": f"alert_{alert_id}",
                    "location": state.upper(),
                    "source_type": "alert",
                    "headline": props.get("event", "Weather Alert"),
                    "narrative_text": narrative,
                    "issued_at": props.get("sent") or props.get("effective"),
                    "payload": props,
                }
            )
        return documents

    def get_forecast_by_point(
        self, lat: float, lon: float, location_name: str
    ) -> list[dict]:
        """Resolve lat/lon to NWS gridpoint forecast and return narrative daily periods."""
        try:
            point_data = self._get(f"{NWS_BASE_URL}/points/{lat},{lon}")
            forecast_url = point_data.get("properties", {}).get("forecast")
            if not forecast_url:
                return []

            forecast_data = self._get(forecast_url)
            periods = forecast_data.get("properties", {}).get("periods", [])

            documents = []
            for p in periods:
                period_name = p.get("name", "Forecast Period")
                detailed_forecast = p.get("detailedForecast", "")
                if not detailed_forecast:
                    continue

                issued = p.get("startTime")
                raw_hash = f"{location_name}_{period_name}_{issued}"
                doc_id = f"forecast_{hashlib.md5(raw_hash.encode()).hexdigest()}"

                documents.append(
                    {
                        "id": doc_id,
                        "location": location_name,
                        "source_type": "forecast",
                        "headline": f"{location_name} - {period_name}",
                        "narrative_text": detailed_forecast,
                        "issued_at": issued,
                        "payload": p,
                    }
                )
            return documents
        except Exception as e:
            logger.warning(
                f"Failed to fetch forecast for {location_name} ({lat},{lon}): {e}"
            )
            return []

    def fetch_weather_documents(self, locations: list[str]) -> list[dict]:
        """Convenience method to harvest alerts and forecasts for a list of locations."""
        all_docs = []
        states_processed = set()

        for loc in locations:
            coords = self.get_coordinates(loc)
            if coords:
                lat, lon = coords
                forecast_docs = self.get_forecast_by_point(lat, lon, loc)
                all_docs.extend(forecast_docs)

            # Extract 2-letter state code if present (e.g., "Chicago, IL" -> "IL")
            if "," in loc:
                state_code = loc.split(",")[-1].strip()
                if len(state_code) == 2 and state_code not in states_processed:
                    alert_docs = self.get_alerts_by_area(state_code)
                    all_docs.extend(alert_docs)
                    states_processed.add(state_code)

        return all_docs