from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import risk, health

# FastAPI app instance — this is the central object that wires
# everything together, similar to how Rails has Application
app = FastAPI(
    title="Supply Chain Disruption Predictor API",
    description=(
        "Real-time supply chain risk scoring API. "
        "Ingests news, weather, and market data to predict "
        "disruption risk using NLP and ML models."
    ),
    version="1.0.0",
)

# CORS middleware — allows the Streamlit dashboard (running on a
# different port) to make requests to this API without being blocked
# by the browser's same-origin policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules — each file in routes/ handles a group of endpoints
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(risk.router, prefix="/api", tags=["Risk"])


@app.on_event("startup")
async def startup_event():
    logger.info("Supply Chain Predictor API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Supply Chain Predictor API shutting down")
