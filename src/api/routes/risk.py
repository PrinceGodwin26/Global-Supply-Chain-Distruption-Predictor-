import math
from datetime import datetime, UTC
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


# ── Response models ────────────────────────────────────────────────────────
# Pydantic models define the exact JSON shape our endpoints return.
# FastAPI validates every response against these automatically.

class RiskLevel(BaseModel):
    level: str          # "Low", "Medium", or "High"
    score: float        # 0.0 to 1.0
    description: str    # human-readable explanation


class CurrentRiskResponse(BaseModel):
    risk: RiskLevel
    signals: dict       # breakdown of contributing signals
    window_start: str
    window_end: str
    generated_at: str


class ArticleRiskResponse(BaseModel):
    id: int
    title: Optional[str]
    source: Optional[str]
    category: str
    risk_score: float
    sentiment_label: str
    published_at: Optional[str]


class RiskTrendResponse(BaseModel):
    windows: list[dict]  # list of recent feature vectors
    trend: str           # "increasing", "decreasing", or "stable"
    avg_risk: float


# ── Helper functions ────────────────────────────────────────────────────────

def get_engine():
    """Creates a SQLAlchemy engine from environment variables."""
    return create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )


def risk_level_from_score(score: float) -> tuple[str, str]:
    """
    Converts a numeric risk score to a human-readable level and description.

    Args:
        score: float between 0 and 1

    Returns:
        Tuple of (level, description)
    """
    if score < 0.3:
        return "Low", "Supply chain conditions appear normal. No significant disruption signals detected."
    elif score < 0.6:
        return "Medium", "Moderate disruption signals detected. Monitor situation closely."
    else:
        return "High", "Significant disruption risk detected. Immediate attention recommended."


def safe_float(value, default: float = 0.0) -> float:
    """
    Converts a value to float, replacing None and NaN with a safe default.

    JSON does not support NaN or Infinity — Python's json module raises
    ValueError when encountering them. This guard ensures all numeric
    values we return are valid JSON floats.

    Args:
        value: the value to convert
        default: fallback value for None/NaN (default 0.0)

    Returns:
        A valid finite float
    """
    try:
        result = float(value)
        return default if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError):
        return default


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/risk/current", response_model=CurrentRiskResponse)
async def get_current_risk():
    """
    Returns the most recent disruption risk assessment.

    Fetches the latest feature vector from the database and returns
    a structured risk assessment including the overall score, risk level,
    and a breakdown of contributing signals (news, weather, market).

    This is the primary endpoint — the one the dashboard calls every
    few minutes to update the risk display.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                window_start, window_end,
                avg_news_risk, max_news_risk, high_risk_article_count,
                avg_wind_speed, severe_weather_port_count,
                avg_oil_change, avg_shipping_stock_change, market_stress_score,
                disruption_risk_index
            FROM feature_vectors
            ORDER BY window_end DESC
            LIMIT 1
        """))
        row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="No risk assessments available yet. Run the scheduler first."
        )

    score = safe_float(row.disruption_risk_index)
    level, description = risk_level_from_score(score)

    return CurrentRiskResponse(
        risk=RiskLevel(level=level, score=score, description=description),
        signals={
            "news": {
                "avg_risk": safe_float(row.avg_news_risk),
                "max_risk": safe_float(row.max_news_risk),
                "high_risk_articles": int(row.high_risk_article_count or 0),
            },
            "weather": {
                "avg_wind_speed_ms": safe_float(row.avg_wind_speed),
                "severe_ports": int(row.severe_weather_port_count or 0),
            },
            "market": {
                "avg_oil_change_pct": safe_float(row.avg_oil_change),
                "avg_shipping_stock_change_pct": safe_float(row.avg_shipping_stock_change),
                "stress_score": safe_float(row.market_stress_score),
            },
        },
        window_start=str(row.window_start),
        window_end=str(row.window_end),
        generated_at=datetime.now(UTC).isoformat(),
    )


@router.get("/risk/articles", response_model=list[ArticleRiskResponse])
async def get_high_risk_articles(
    limit: int = Query(default=20, ge=1, le=100),
    min_score: float = Query(default=0.6, ge=0.0, le=1.0),
):
    """
    Returns the highest-risk news articles above a minimum score threshold.

    Args:
        limit: maximum number of articles to return (1-100, default 20)
        min_score: minimum risk score to include (0.0-1.0, default 0.6)

    Query parameters can be adjusted in the URL:
        /api/risk/articles?limit=10&min_score=0.8
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, title, source, category, risk_score,
                   sentiment_label, published_at
            FROM news_articles
            WHERE risk_score >= :min_score
            AND sentiment_label != 'IRRELEVANT'
            ORDER BY risk_score DESC
            LIMIT :limit
        """), {"min_score": min_score, "limit": limit})
        rows = result.fetchall()

    return [
        ArticleRiskResponse(
            id=row.id,
            title=row.title,
            source=row.source,
            category=row.category,
            risk_score=float(row.risk_score),
            sentiment_label=row.sentiment_label,
            published_at=str(row.published_at) if row.published_at else None,
        )
        for row in rows
    ]


@router.get("/risk/trend", response_model=RiskTrendResponse)
async def get_risk_trend(
    windows: int = Query(default=24, ge=2, le=168),
):
    """
    Returns disruption risk trend over the last N collection windows.

    Args:
        windows: number of recent windows to include (2-168, default 24 = 12 hours)

    Trend is calculated by comparing the average risk of the first half
    of the window set against the second half:
    - "increasing": second half avg > first half avg by >0.05
    - "decreasing": second half avg < first half avg by >0.05
    - "stable": difference within ±0.05
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT window_start, window_end, disruption_risk_index,
                   avg_news_risk, market_stress_score
            FROM feature_vectors
            ORDER BY window_end DESC
            LIMIT :windows
        """), {"windows": windows})
        rows = result.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No trend data available yet."
        )

    # Reverse to chronological order for trend analysis
    rows = list(reversed(rows))

    risk_scores = [safe_float(r.disruption_risk_index) for r in rows]
    avg_risk = sum(risk_scores) / len(risk_scores)

    # Compare first half vs second half to determine trend direction
    mid = len(risk_scores) // 2
    first_half_avg = sum(risk_scores[:mid]) / max(len(risk_scores[:mid]), 1)
    second_half_avg = sum(risk_scores[mid:]) / max(len(risk_scores[mid:]), 1)
    diff = second_half_avg - first_half_avg

    if diff > 0.05:
        trend = "increasing"
    elif diff < -0.05:
        trend = "decreasing"
    else:
        trend = "stable"

    return RiskTrendResponse(
        windows=[
            {
                "window_start": str(r.window_start),
                "window_end": str(r.window_end),
                "risk_score": safe_float(r.disruption_risk_index),
                "news_risk": safe_float(r.avg_news_risk),
                "market_stress": safe_float(r.market_stress_score),
            }
            for r in rows
        ],
        trend=trend,
        avg_risk=round(avg_risk, 4),
    )
