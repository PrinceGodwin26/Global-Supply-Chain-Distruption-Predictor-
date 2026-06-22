from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

from src.ingestion.news_collector import NewsCollector
from src.ingestion.weather_collector import WeatherCollector
from src.ingestion.market_collector import MarketCollector


def run_all_collectors():
    """
    Runs all three data collectors in sequence: news, weather, and market data.

    Each collector is wrapped in its own try/except block. This is an
    important design choice — if, say, NewsAPI is temporarily down, we
    still want weather and market data to be collected successfully.
    One source failing should never block the others.
    """
    logger.info("=" * 60)
    logger.info("Starting scheduled data collection run")
    logger.info("=" * 60)

    # --- News ---
    try:
        news_collector = NewsCollector()
        articles = news_collector.fetch()
        news_collector.save_to_db(articles)
    except Exception as e:
        logger.error(f"News collection failed: {e}")

    # --- Weather ---
    try:
        weather_collector = WeatherCollector()
        weather_data = weather_collector.fetch_all()
        weather_collector.save_to_db(weather_data)
    except Exception as e:
        logger.error(f"Weather collection failed: {e}")

    # --- Market ---
    try:
        market_collector = MarketCollector()
        market_data = market_collector.fetch_all()
        market_collector.save_to_db(market_data)
    except Exception as e:
        logger.error(f"Market collection failed: {e}")

    logger.info("Scheduled data collection run complete")


if __name__ == "__main__":
    # BlockingScheduler takes over the current thread entirely — this
    # script is meant to run continuously in the background, not return
    # control until manually stopped (Ctrl+C)
    scheduler = BlockingScheduler()

    # Run once immediately on startup, so we don't have to wait 30 minutes
    # to see the first results
    run_all_collectors()

    # Then schedule it to repeat every 30 minutes indefinitely
    scheduler.add_job(run_all_collectors, "interval", minutes=30)

    logger.info("Scheduler started — collecting data every 30 minutes. Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
