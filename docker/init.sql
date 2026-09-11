-- This script runs automatically when the PostgreSQL container
-- starts for the first time (via /docker-entrypoint-initdb.d/)
-- It creates all tables so the app works immediately without
-- any manual psql setup

CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    source TEXT,
    description TEXT,
    category TEXT,
    published_at TIMESTAMPTZ,
    url TEXT UNIQUE,
    fetched_at TIMESTAMPTZ NOT NULL,
    risk_score FLOAT,
    sentiment_label TEXT,
    processed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS weather_snapshots (
    id SERIAL PRIMARY KEY,
    port_name TEXT,
    latitude FLOAT,
    longitude FLOAT,
    temperature_c FLOAT,
    weather_condition TEXT,
    weather_description TEXT,
    wind_speed_ms FLOAT,
    humidity_percent INTEGER,
    fetched_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id SERIAL PRIMARY KEY,
    symbol TEXT,
    label TEXT,
    latest_close FLOAT,
    previous_close FLOAT,
    percent_change FLOAT,
    fetched_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_vectors (
    id SERIAL PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    avg_news_risk FLOAT,
    max_news_risk FLOAT,
    high_risk_article_count INTEGER,
    geopolitical_count INTEGER,
    natural_disaster_count INTEGER,
    labor_dispute_count INTEGER,
    trade_policy_count INTEGER,
    logistics_count INTEGER,
    avg_wind_speed FLOAT,
    max_wind_speed FLOAT,
    severe_weather_port_count INTEGER,
    avg_oil_change FLOAT,
    avg_shipping_stock_change FLOAT,
    market_stress_score FLOAT,
    disruption_risk_index FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
