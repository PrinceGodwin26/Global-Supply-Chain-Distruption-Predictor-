import os
from datetime import datetime, UTC
import requests
from dotenv import load_dotenv
from loguru import logger
from src.utils.database import get_connection

load_dotenv()


class WeatherCollector:
    """
    Fetches current weather conditions for major global shipping ports.

    Severe weather at key ports is a leading indicator of supply chain
    disruption — storms, high winds, and extreme temperatures can delay
    cargo loading/unloading and vessel movement.
    """

    # Major shipping ports we track — (display name, latitude, longitude)
    # Coordinates are used instead of city names because they're unambiguous
    # (multiple cities worldwide share the same name)
    PORTS = [
        {"name": "Shanghai", "lat": 31.2304, "lon": 121.4737},
        {"name": "Rotterdam", "lat": 51.9244, "lon": 4.4777},
        {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
        {"name": "Mumbai", "lat": 18.9388, "lon": 72.8354},
        {"name": "Los Angeles", "lat": 33.7406, "lon": -118.2706},
    ]

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = os.getenv("OPENWEATHER_BASE_URL")
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY not found in environment variables")
        if not self.base_url:
            raise ValueError("OPENWEATHER_BASE_URL not found in environment variables")

    def fetch_for_port(self, port: dict) -> dict | None:
        """
        Fetch current weather for a single port location.

        Args:
            port: a dict with 'name', 'lat', 'lon' keys

        Returns:
            A standardized weather dict, or None if the request failed
        """
        params = {
            "lat": port["lat"],
            "lon": port["lon"],
            "appid": self.api_key,
            "units": "metric",  # Celsius instead of Kelvin
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather for {port['name']}: {e}")
            return None

        data = response.json()

        # Flatten the nested response into a simple, flat structure —
        # easier to store in a database row or pandas DataFrame later
        return {
            "port_name": port["name"],
            "latitude": port["lat"],
            "longitude": port["lon"],
            "temperature_c": data.get("main", {}).get("temp"),
            "weather_condition": data.get("weather", [{}])[0].get("main"),
            "weather_description": data.get("weather", [{}])[0].get("description"),
            "wind_speed_ms": data.get("wind", {}).get("speed"),
            "humidity_percent": data.get("main", {}).get("humidity"),
            "fetched_at": datetime.now(UTC).isoformat(),
        }


    def save_to_db(self, weather_data: list[dict]) -> int:
        """
        Inserts fetched weather snapshots into the weather_snapshots table.

        Unlike news articles, weather snapshots don't have a natural unique
        key to deduplicate on — each fetch represents a new point-in-time
        reading, even if the port and conditions happen to repeat. So we
        simply insert every row; duplicates here are meaningful data points
        (e.g. "still raining 30 minutes later"), not errors.

        Args:
            weather_data: list of weather dicts, as returned by fetch_all()

        Returns:
            The number of rows inserted
        """
        if not weather_data:
            logger.info("No weather data to save")
            return 0

        conn = get_connection()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO weather_snapshots
                (port_name, latitude, longitude, temperature_c, weather_condition,
                 weather_description, wind_speed_ms, humidity_percent, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for w in weather_data:
            cursor.execute(insert_query, (
                w.get("port_name"),
                w.get("latitude"),
                w.get("longitude"),
                w.get("temperature_c"),
                w.get("weather_condition"),
                w.get("weather_description"),
                w.get("wind_speed_ms"),
                w.get("humidity_percent"),
                w.get("fetched_at"),
            ))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Inserted {len(weather_data)} weather snapshots")
        return len(weather_data)

    def fetch_all(self) -> list[dict]:
        """
        Fetch current weather for every tracked port.

        Returns:
            A list of weather dicts, one per port (skipping any that failed)
        """
        logger.info(f"Fetching weather for {len(self.PORTS)} ports")

        results = []
        for port in self.PORTS:
            weather = self.fetch_for_port(port)
            if weather:
                results.append(weather)

        logger.info(f"Successfully fetched weather for {len(results)}/{len(self.PORTS)} ports")
        return results


if __name__ == "__main__":
    collector = WeatherCollector()
    weather_data = collector.fetch_all()
    collector.save_to_db(weather_data)

    print(f"\nWeather for {len(weather_data)} ports:\n")
    for w in weather_data:
        print(f"- {w['port_name']}: {w['temperature_c']}°C, {w['weather_description']}, wind {w['wind_speed_ms']} m/s")