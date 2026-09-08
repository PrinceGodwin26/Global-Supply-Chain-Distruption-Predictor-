import math
from datetime import datetime, UTC, timedelta
from loguru import logger

from src.utils.database import get_connection


def safe_float(value, default: float = 0.0) -> float:
    """
    Converts a value to a safe float, replacing None/NaN/Inf with default.
    Prevents NaN values from corrupting the feature vector and disruption index.
    """
    try:
        result = float(value)
        return default if (math.isnan(result) or math.isinf(result)) else result
    except (TypeError, ValueError):
        return default


# Severe weather conditions from OpenWeatherMap that indicate
# meaningful disruption risk at a port location
SEVERE_WEATHER_CONDITIONS = {
    "Thunderstorm", "Drizzle", "Rain", "Snow", "Mist",
    "Smoke", "Haze", "Dust", "Fog", "Sand", "Ash",
    "Squall", "Tornado",
}

# Weights for combining signals into the final disruption risk index.
# News carries the most signal because it directly describes events.
# Market reacts quickly to real disruptions (prices move fast).
# Weather is real but slower-moving and more localized.
NEWS_WEIGHT = 0.50
MARKET_WEIGHT = 0.30
WEATHER_WEIGHT = 0.20


class FeatureEngineer:
    """
    Aggregates raw collected data into feature vectors for ML model training.

    Each feature vector represents one 30-minute collection window,
    summarizing the state of news risk, weather conditions, and market
    signals into a fixed-length numeric row.

    Over time, these rows form a time series that the LSTM model in
    Phase 5 learns sequential patterns from — e.g. "risk score has been
    climbing for 3 consecutive windows" is a stronger signal than any
    single window in isolation.
    """

    def __init__(self, window_minutes: int = 30):
        # window_minutes defines how far back we look when aggregating data
        # for each feature vector — matches our scheduler interval
        self.window_minutes = window_minutes

    def _compute_news_features(self, cursor, window_start: datetime, window_end: datetime) -> dict:
        """
        Aggregates news article scores within the time window.

        Args:
            cursor: active database cursor
            window_start: start of the collection window
            window_end: end of the collection window

        Returns:
            Dict of news-derived features
        """
        cursor.execute("""
            SELECT
                AVG(risk_score)                                    AS avg_risk,
                MAX(risk_score)                                    AS max_risk,
                COUNT(*) FILTER (WHERE risk_score > 0.7)          AS high_risk_count,
                COUNT(*) FILTER (WHERE category = 'geopolitical') AS geo_count,
                COUNT(*) FILTER (WHERE category = 'natural_disaster') AS nat_count,
                COUNT(*) FILTER (WHERE category = 'labor_dispute')    AS lab_count,
                COUNT(*) FILTER (WHERE category = 'trade_policy')     AS trade_count,
                COUNT(*) FILTER (WHERE category = 'logistics')        AS log_count
            FROM news_articles
            WHERE fetched_at BETWEEN %s AND %s
            AND risk_score IS NOT NULL
            AND sentiment_label != 'IRRELEVANT'
        """, (window_start, window_end))

        row = cursor.fetchone()

        # Handle case where no articles exist in this window
        if not row or row[0] is None:
            logger.warning("No scored news articles found in window — using zero defaults")
            return {
                "avg_news_risk": 0.0,
                "max_news_risk": 0.0,
                "high_risk_article_count": 0,
                "geopolitical_count": 0,
                "natural_disaster_count": 0,
                "labor_dispute_count": 0,
                "trade_policy_count": 0,
                "logistics_count": 0,
            }

        return {
            "avg_news_risk": round(float(row[0] or 0), 4),
            "max_news_risk": round(float(row[1] or 0), 4),
            "high_risk_article_count": int(row[2] or 0),
            "geopolitical_count": int(row[3] or 0),
            "natural_disaster_count": int(row[4] or 0),
            "labor_dispute_count": int(row[5] or 0),
            "trade_policy_count": int(row[6] or 0),
            "logistics_count": int(row[7] or 0),
        }

    def _compute_weather_features(self, cursor, window_start: datetime, window_end: datetime) -> dict:
        """
        Aggregates weather conditions across all tracked ports within the window.

        Args:
            cursor: active database cursor
            window_start: start of the collection window
            window_end: end of the collection window

        Returns:
            Dict of weather-derived features
        """
        cursor.execute("""
            SELECT
                AVG(wind_speed_ms)      AS avg_wind,
                MAX(wind_speed_ms)      AS max_wind,
                weather_condition
            FROM weather_snapshots
            WHERE fetched_at BETWEEN %s AND %s
            GROUP BY weather_condition
        """, (window_start, window_end))

        rows = cursor.fetchall()

        if not rows:
            logger.warning("No weather data found in window — using zero defaults")
            return {
                "avg_wind_speed": 0.0,
                "max_wind_speed": 0.0,
                "severe_weather_port_count": 0,
            }

        avg_winds = [r[0] for r in rows if r[0] is not None]
        max_winds = [r[1] for r in rows if r[1] is not None]
        conditions = [r[2] for r in rows if r[2] is not None]

        severe_count = sum(
            1 for c in conditions if c in SEVERE_WEATHER_CONDITIONS
        )

        return {
            "avg_wind_speed": round(safe_float(sum(avg_winds) / len(avg_winds)) if avg_winds else 0.0, 4),
            "max_wind_speed": round(safe_float(max(max_winds)) if max_winds else 0.0, 4),
            "severe_weather_port_count": severe_count,
        }

    def _compute_market_features(self, cursor, window_start: datetime, window_end: datetime) -> dict:
        """
        Aggregates market signals within the window.

        Oil price changes (Brent + WTI) and shipping stock movements
        (ZIM + MATX) are averaged separately, then combined into a
        market stress score that increases when either oil spikes
        (supply disruption) or shipping stocks fall (sector distress).

        Args:
            cursor: active database cursor
            window_start: start of the collection window
            window_end: end of the collection window

        Returns:
            Dict of market-derived features
        """
        cursor.execute("""
            SELECT symbol, AVG(percent_change) AS avg_change
            FROM market_snapshots
            WHERE fetched_at BETWEEN %s AND %s
            GROUP BY symbol
        """, (window_start, window_end))

        rows = cursor.fetchall()

        if not rows:
            logger.warning("No market data found in window — using zero defaults")
            return {
                "avg_oil_change": 0.0,
                "avg_shipping_stock_change": 0.0,
                "market_stress_score": 0.0,
            }

        symbol_changes = {row[0]: float(row[1]) for row in rows}

        # Average oil price change (Brent + WTI)
        oil_changes = [
            symbol_changes.get("BZ=F", 0),
            symbol_changes.get("CL=F", 0),
        ]
        avg_oil = sum(oil_changes) / len(oil_changes)

        # Average shipping stock change (ZIM + MATX)
        shipping_changes = [
            symbol_changes.get("ZIM", 0),
            symbol_changes.get("MATX", 0),
        ]
        avg_shipping = sum(shipping_changes) / len(shipping_changes)

        # Market stress score logic:
        # - Oil prices rising sharply = supply disruption signal (positive change = stress)
        # - Shipping stocks falling = sector distress signal (negative change = stress)
        # We normalize each to [0, 1] assuming max meaningful change of ±10%
        oil_stress = min(max(avg_oil / 10, 0), 1)
        shipping_stress = min(max(-avg_shipping / 10, 0), 1)
        market_stress = round((oil_stress + shipping_stress) / 2, 4)

        return {
            "avg_oil_change": round(safe_float(avg_oil), 4),
            "avg_shipping_stock_change": round(safe_float(avg_shipping), 4),
            "market_stress_score": safe_float(market_stress),
        }

    def _compute_disruption_risk_index(
        self,
        news_features: dict,
        weather_features: dict,
        market_features: dict,
    ) -> float:
        """
        Combines news, weather, and market signals into a single
        disruption risk index between 0 and 1.

        Weighted combination:
            news   × 0.50  (most direct signal — articles describe events)
            market × 0.30  (fast-reacting proxy — prices move before news)
            weather × 0.20 (real but slower-moving and localized)

        Weather contribution is normalized: severe port count / 5 ports
        (since we track 5 ports, 5/5 severe = maximum weather stress).

        Returns:
            A float between 0.0 and 1.0
        """
        news_signal = news_features["avg_news_risk"]

        # Normalize severe weather port count to 0-1 scale
        weather_signal = min(
            weather_features["severe_weather_port_count"] / 5, 1.0
        )

        market_signal = market_features["market_stress_score"]

        # Ensure all signals are clean floats before arithmetic
        news_signal = safe_float(news_signal)
        weather_signal = safe_float(weather_signal)
        market_signal = safe_float(market_signal)

        index = (
            news_signal * NEWS_WEIGHT
            + market_signal * MARKET_WEIGHT
            + weather_signal * WEATHER_WEIGHT
        )

        # Clamp to [0, 1] for safety
        final = safe_float(index)
        return round(max(0.0, min(1.0, final)), 4)

    def generate_feature_vector(self, window_end: datetime | None = None) -> dict | None:
        """
        Generates a single feature vector for the most recent window.

        Args:
            window_end: end of the window (defaults to now)

        Returns:
            The feature vector dict if successful, None if an error occurred
        """
        if window_end is None:
            window_end = datetime.now(UTC)

        window_start = window_end - timedelta(minutes=self.window_minutes)

        logger.info(f"Generating feature vector for window: {window_start} → {window_end}")

        conn = get_connection()
        cursor = conn.cursor()

        try:
            news_features = self._compute_news_features(cursor, window_start, window_end)
            weather_features = self._compute_weather_features(cursor, window_start, window_end)
            market_features = self._compute_market_features(cursor, window_start, window_end)

            disruption_index = self._compute_disruption_risk_index(
                news_features, weather_features, market_features
            )

            feature_vector = {
                "window_start": window_start,
                "window_end": window_end,
                **news_features,
                **weather_features,
                **market_features,
                "disruption_risk_index": disruption_index,
            }

            # Save to database
            cursor.execute("""
                INSERT INTO feature_vectors (
                    window_start, window_end,
                    avg_news_risk, max_news_risk, high_risk_article_count,
                    geopolitical_count, natural_disaster_count,
                    labor_dispute_count, trade_policy_count, logistics_count,
                    avg_wind_speed, max_wind_speed, severe_weather_port_count,
                    avg_oil_change, avg_shipping_stock_change, market_stress_score,
                    disruption_risk_index
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                feature_vector["window_start"],
                feature_vector["window_end"],
                feature_vector["avg_news_risk"],
                feature_vector["max_news_risk"],
                feature_vector["high_risk_article_count"],
                feature_vector["geopolitical_count"],
                feature_vector["natural_disaster_count"],
                feature_vector["labor_dispute_count"],
                feature_vector["trade_policy_count"],
                feature_vector["logistics_count"],
                feature_vector["avg_wind_speed"],
                feature_vector["max_wind_speed"],
                feature_vector["severe_weather_port_count"],
                feature_vector["avg_oil_change"],
                feature_vector["avg_shipping_stock_change"],
                feature_vector["market_stress_score"],
                feature_vector["disruption_risk_index"],
            ))

            conn.commit()
            logger.info(f"Feature vector saved — disruption_risk_index: {disruption_index}")
            return feature_vector

        except Exception as e:
            logger.error(f"Failed to generate feature vector: {e}")
            conn.rollback()
            return None

        finally:
            # finally block always runs — ensures connection is always
            # closed even if an exception occurred above
            cursor.close()
            conn.close()


if __name__ == "__main__":
    engineer = FeatureEngineer()
    fv = engineer.generate_feature_vector()

    if fv:
        print("\nGenerated feature vector:\n")
        for key, value in fv.items():
            if key not in ("window_start", "window_end"):
                print(f"  {key:<35} {value}")
        print(f"\n  Disruption Risk Index: {fv['disruption_risk_index']}")
