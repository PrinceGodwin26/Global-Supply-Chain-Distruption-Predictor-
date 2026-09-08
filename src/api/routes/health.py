from datetime import datetime, UTC
from fastapi import APIRouter
from pydantic import BaseModel
from src.utils.database import get_connection

router = APIRouter()


class HealthResponse(BaseModel):
    """
    Pydantic model defining the exact shape of our health check response.
    FastAPI uses this to validate the response and generate documentation.
    """
    status: str
    timestamp: str
    database: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint — confirms the API and database are reachable.

    Used by:
    - Docker health checks to know if the container is ready
    - Monitoring systems to alert if the service goes down
    - Load balancers to route traffic only to healthy instances

    Returns 200 OK with database status when everything is working.
    """
    db_status = "connected"
    try:
        conn = get_connection()
        conn.close()
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="ok",
        timestamp=datetime.now(UTC).isoformat(),
        database=db_status,
        version="1.0.0",
    )
