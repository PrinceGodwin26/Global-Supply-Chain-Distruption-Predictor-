import os
from datetime import datetime, UTC
import yfinance as yf
from loguru import logger


class MarketCollector:
    """
    Fetches market data relevant to supply chain health — oil prices and
    shipping freight indices act as real-time, market-driven signals of
    disruption, often reacting faster than news coverage.
    """

    # Yahoo Finance ticker symbols we track, with human-readable labels
    TICKERS = {
        "BZ=F": "Brent Crude Oil",
        "CL=F": "WTI Crude Oil",
        "ZIM": "ZIM Integrated Shipping",
        "MATX": "Matson Inc (Shipping)",
    }

    def fetch_for_ticker(self, symbol: str, label: str) -> dict | None:
        """
        Fetch the latest available price data for a single ticker.

        Args:
            symbol: the Yahoo Finance ticker symbol (e.g. 'BZ=F')
            label: a human-readable name for logging/storage

        Returns:
            A standardized dict with price info, or None if fetch failed
        """
        try:
            ticker = yf.Ticker(symbol)
            # period="2d" gets the last 2 trading days so we can calculate
            # a day-over-day change, not just a single snapshot value
            history = ticker.history(period="2d")

            if history.empty:
                logger.warning(f"No data returned for {label} ({symbol})")
                return None

            latest = history.iloc[-1]
            previous = history.iloc[-2] if len(history) > 1 else latest

            latest_close = float(latest["Close"])
            previous_close = float(previous["Close"])
            percent_change = ((latest_close - previous_close) / previous_close) * 100

            return {
                "symbol": symbol,
                "label": label,
                "latest_close": round(latest_close, 2),
                "previous_close": round(previous_close, 2),
                "percent_change": round(percent_change, 2),
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            # yfinance can raise several different exception types depending
            # on network issues vs invalid symbols — catching broadly here
            # and logging keeps one bad ticker from crashing the whole batch
            logger.error(f"Failed to fetch {label} ({symbol}): {e}")
            return None

    def fetch_all(self) -> list[dict]:
        """
        Fetch market data for every tracked ticker.

        Returns:
            A list of market data dicts, one per ticker (skipping failures)
        """
        logger.info(f"Fetching market data for {len(self.TICKERS)} tickers")

        results = []
        for symbol, label in self.TICKERS.items():
            data = self.fetch_for_ticker(symbol, label)
            if data:
                results.append(data)

        logger.info(f"Successfully fetched {len(results)}/{len(self.TICKERS)} tickers")
        return results


if __name__ == "__main__":
    collector = MarketCollector()
    market_data = collector.fetch_all()

    print(f"\nMarket data for {len(market_data)} tickers:\n")
    for m in market_data:
        direction = "UP" if m["percent_change"] >= 0 else "DOWN"
        print(f"- {m['label']}: ${m['latest_close']} ({direction} {abs(m['percent_change'])}%)")
