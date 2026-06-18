import os
from datetime import datetime, UTC
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class NewsCollector:
    """
    Fetches news articles related to supply chain disruptions from NewsAPI.

    This class wraps NewsAPI's /everything endpoint and provides a clean
    interface for fetching, validating, and returning structured article data.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            # Fail loudly and early if the key is missing — this prevents
            # confusing downstream errors later in the pipeline
            raise ValueError("NEWS_API_KEY not found in environment variables")

    def fetch(self, query: str = "supply chain disruption", page_size: int = 20) -> list[dict]:
        """
        Fetch articles matching the given query.

        Args:
            query: search keywords sent to NewsAPI
            page_size: number of articles to retrieve (max 100 per NewsAPI's free tier)

        Returns:
            A list of dictionaries, each representing one article with
            standardized fields: title, source, description, published_at, url
        """
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",  # newest articles first
        }

        logger.info(f"Fetching news articles for query: '{query}'")

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()  # raises an exception for HTTP errors (4xx, 5xx)
        except requests.exceptions.RequestException as e:
            # Catching the broad exception here so a network hiccup doesn't
            # crash the whole pipeline — we log it and return an empty list instead
            logger.error(f"Failed to fetch news: {e}")
            return []

        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"NewsAPI returned an error: {data.get('message')}")
            return []

        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} articles")

        # Standardize the structure — NewsAPI's raw response has nested
        # objects (like source.name) which we flatten here for easier
        # downstream processing (e.g. saving to a database or CSV)
        cleaned_articles = []
        for article in articles:
            cleaned_articles.append({
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
                "description": article.get("description"),
                "published_at": article.get("publishedAt"),
                "url": article.get("url"),
                "fetched_at": datetime.now(UTC).isoformat(),
            })

        return cleaned_articles


# This block only runs when you execute this file directly
# (python3 src/ingestion/news_collector.py) — NOT when it's imported
# elsewhere in the project. This is the standard Python pattern for
# adding a quick self-test to any module.
if __name__ == "__main__":
    collector = NewsCollector()
    articles = collector.fetch()

    print(f"\nFetched {len(articles)} articles:\n")
    for article in articles[:5]:
        print(f"- {article['title']} ({article['source']})")