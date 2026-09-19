# Supply Chain Disruption Predictor

A real-time machine learning system that monitors global supply chain risk by ingesting live news, weather, and market data — scoring each signal using NLP and combining them into a unified, explainable disruption risk index.

Built as an end-to-end ML engineering portfolio project covering the full lifecycle: data ingestion → NLP processing → feature engineering → model training → REST API → interactive dashboard — all containerised with Docker Compose.

---

## What it does

Every 30 minutes, the system automatically:

1. Pulls live news from NewsAPI across 5 disruption categories (geopolitical conflict, natural disasters, labor disputes, trade policy, logistics)
2. Fetches weather conditions at 5 major global shipping ports (Shanghai, Rotterdam, Singapore, Mumbai, Los Angeles)
3. Retrieves oil price and shipping stock data from Yahoo Finance (Brent, WTI, ZIM, MATX)
4. Scores each news article using a 3-step NLP pipeline (relevance filter → DistilBERT sentiment → disruption signal verification)
5. Aggregates all signals into a 15-column feature vector with a single `disruption_risk_index` (0–1)
6. Serves predictions via a FastAPI REST API and visualises them on a live Streamlit dashboard

---

## Real-world context

This system mirrors the architecture used by commercial supply chain intelligence platforms (Resilinc, Everstream Analytics) that sell risk monitoring to Fortune 500 companies. During development, the system independently caught and scored the Strait of Hormuz crisis (Iran-US conflict, 2026) — surfacing articles like *"Ships Stack Up at Hormuz Entrance as US Navy Escorts Resume"* and *"Typhoon Dolphin Deepens China Port Congestion, Stranding 2.4M TEUs"* with risk scores above 0.99.

---

## System architecture

```
┌─────────────────────────────────────────────────────┐
│  Data Sources                                       │
│  NewsAPI · OpenWeatherMap · Yahoo Finance           │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Ingestion Layer  (APScheduler, every 30 min)       │
│  NewsCollector · WeatherCollector · MarketCollector │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Processing Layer                                   │
│  NLP Pipeline (DistilBERT) · Feature Engineering   │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Storage  (PostgreSQL)                              │
│  news_articles · weather_snapshots                  │
│  market_snapshots · feature_vectors                 │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Serving Layer                                      │
│  FastAPI REST API → Streamlit Dashboard             │
└─────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Data ingestion | Python, requests, newsapi-python, yfinance | Live data collection |
| Scheduling | APScheduler | Automated 30-minute collection cycles |
| NLP | HuggingFace Transformers, DistilBERT | Article risk scoring |
| Deep learning | PyTorch (CPU) | LSTM sequence model |
| ML | XGBoost, scikit-learn | Risk classification |
| Database | PostgreSQL 16, psycopg2, SQLAlchemy | Data persistence |
| API | FastAPI, Pydantic, Uvicorn | REST endpoints |
| Dashboard | Streamlit, Plotly | Interactive risk visualisation |
| Infrastructure | Docker, Docker Compose | Containerised deployment |

---

## NLP scoring pipeline

Each news article passes through three stages before a risk score is assigned:

**Stage 1 — Relevance filter**
Checks whether the article title or description contains supply-chain-specific keywords (`"supply chain"`, `"port closure"`, `"shipping delay"`, `"strait"`, `"tariff"`, etc.). Articles that match no keywords receive a score of 0.1 (low risk, not zero — to avoid overconfidence).

**Stage 2 — Sentiment scoring**
Passes the cleaned title + description through `distilbert-base-uncased-finetuned-sst-2-english`. Negative sentiment → higher disruption risk. Positive sentiment → lower risk. An escalation override handles positively-framed disruption articles (e.g. *"Shipping costs soar"* — positive sentiment, but a clear disruption signal).

**Stage 3 — Disruption signal verification**
Articles scoring above 0.9 are checked for explicit disruption-signal words (`"closure"`, `"shutdown"`, `"blocked"`, `"strike"`, `"shortage"`, etc.). Articles missing all signals are capped at 0.75 — preventing false positives from product announcements or financial news that use supply-chain vocabulary without describing an actual disruption.

**Category weighting** is applied after scoring:

| Category | Weight | Rationale |
|---|---|---|
| logistics | 1.00 | Most precise queries — highest trust |
| trade_policy | 0.90 | Directly supply-chain related |
| labor_dispute | 0.85 | Some non-supply-chain strikes |
| natural_disaster | 0.75 | Broader queries, more noise |
| geopolitical | 0.75 | Many non-supply-chain political articles |

---

## Disruption risk index

The final `disruption_risk_index` (0–1) for each 30-minute window combines:

```
disruption_risk_index =
    avg_news_risk          × 0.50   ← dominant signal
  + market_stress_score    × 0.30   ← fast-reacting proxy
  + weather_stress_score   × 0.20   ← slower, localised
```

Risk levels: **Low** < 0.3 · **Medium** 0.3–0.6 · **High** > 0.6

---

## ML models

### XGBoost classifier
Classifies current risk windows as Low / Medium / High based on the 14 engineered features. Feature importance (from training):

- `avg_news_risk` — 41% of decisions
- `high_risk_article_count` — 33%
- `max_news_risk` — 12%
- Weather and market signals — remainder

### LSTM predictor
Learns sequential patterns across 12 consecutive windows (6 hours of history) to predict whether risk is trending upward — enabling early warning before a disruption becomes obvious from current data alone. Requires 50+ feature vectors to train meaningfully.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System health — API + database status |
| GET | `/api/risk/current` | Latest risk score with signal breakdown |
| GET | `/api/risk/articles` | High-risk articles (filterable by score threshold) |
| GET | `/api/risk/trend` | Risk trend over last N windows |

Interactive docs available at `http://localhost:8000/docs` when running.

---

## Quick start

### Prerequisites
- Docker Desktop
- NewsAPI key (free at [newsapi.org](https://newsapi.org/register))
- OpenWeatherMap key (free at [openweathermap.org](https://openweathermap.org/api))

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/PrinceGodwin26/supply-chain-predictor.git
cd supply-chain-predictor

# 2. Create your environment file
cp .env.example .env
# Edit .env and add your API keys and database credentials

# 3. Start the entire system with one command
docker-compose up --build
```

That's it. Docker Compose starts the database, scheduler, API, and dashboard together.

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

---

## Project structure

```
supply-chain-predictor/
├── src/
│   ├── ingestion/
│   │   ├── news_collector.py       # NewsAPI — 5 disruption categories
│   │   ├── weather_collector.py    # OpenWeatherMap — 5 global ports
│   │   ├── market_collector.py     # Yahoo Finance — oil + shipping stocks
│   │   └── scheduler.py           # APScheduler orchestrator
│   ├── processing/
│   │   ├── text_cleaner.py        # HTML/URL stripping, truncation
│   │   ├── nlp_processor.py       # DistilBERT pipeline + scoring logic
│   │   └── feature_engineer.py    # Window aggregation → feature vectors
│   ├── models/
│   │   ├── data_loader.py         # DB → numpy/pandas for model training
│   │   ├── xgboost_model.py       # XGBoost classifier
│   │   └── lstm_model.py          # PyTorch LSTM sequence predictor
│   ├── api/
│   │   ├── main.py                # FastAPI app + CORS + lifecycle
│   │   └── routes/
│   │       ├── health.py          # /api/health
│   │       └── risk.py            # /api/risk/* endpoints
│   ├── dashboard/
│   │   └── app.py                 # Streamlit dashboard
│   └── utils/
│       └── database.py            # psycopg2 connection helper
├── docker/
│   └── init.sql                   # Auto-creates all tables on first start
├── data/
│   └── models/                    # Saved XGBoost + LSTM model files
├── logs/                          # Scheduler runtime logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Known limitations

**NLP domain mismatch** — DistilBERT was fine-tuned on movie review sentiment, not supply-chain text. Some false positives exist where general business or financial articles use supply-chain vocabulary without describing an actual disruption. Resolution: fine-tune on a labeled supply-chain corpus.

**LSTM training data** — The LSTM requires 50+ feature vectors (≈25 hours of continuous collection) to train meaningfully. With limited initial data, XGBoost provides the primary classification signal.

**yfinance reliability** — Yahoo Finance's unofficial API occasionally returns stale or empty data, particularly outside market hours. ZIM and MATX shipping stocks are used as a free proxy for the Baltic Dry Index (which is paywalled on all public APIs).

**NewsAPI free tier** — 100 requests/day limit. With 5 categories × 2 requests each per scheduler run = 10 requests per cycle, the free tier supports approximately 4–5 scheduler runs per day.

---

## Author

**Prince** — Backend Developer & ML Engineer
Hyderabad, India · [github.com/PrinceGodwin26](https://github.com/PrinceGodwin26)
