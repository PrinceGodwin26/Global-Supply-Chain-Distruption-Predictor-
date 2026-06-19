import os
from datetime import datetime, UTC
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class NewsCollector:
    """
    Fetches news articles related to supply chain disruption risk from NewsAPI.

    Rather than relying on a single broad query, this collector searches
    across multiple disruption categories — geopolitical conflict, natural
    disasters, labor disputes, trade policy, and logistics failures.
    This catches events that wouldn't contain the literal phrase
    "supply chain disruption" but are highly relevant (e.g. a war affecting
    a major shipping strait, or a port workers' strike).
    """

    # Each category has its own targeted query. NewsAPI does literal keyword
    # matching, so a single vague query misses conceptually-related events
    # that don't use that exact phrasing.
    QUERY_CATEGORIES = {
        "geopolitical": (
            '(war OR conflict OR blockade OR sanctions OR "military action") '
            'AND ("supply chain" OR "shipping route" OR "trade route" OR "global trade" OR "export ban")'
        ),
        "natural_disaster": (
            '(earthquake OR hurricane OR typhoon OR flood OR wildfire) '
            'AND ("supply chain" OR "shipping route" OR "factory shutdown" OR "port closure" OR logistics)'
        ),
        "labor_dispute": (
            '(strike OR "labor dispute" OR "worker protest") '
            'AND ("supply chain" OR "port workers" OR "factory workers" OR shipping OR logistics)'
        ),
        "trade_policy": (
            '(tariff OR "trade ban" OR "export restriction") '
            'AND ("supply chain" OR shipping OR import OR export OR manufacturing)'
        ),
        "logistics": (
            '"port congestion" OR "shipping delay" OR "container shortage" '
            'OR "supply chain disruption"'
        ),
    }

    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        self.base_url = os.getenv("NEWS_API_BASE_URL")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        if not self.base_url:
            raise ValueError("NEWS_API_BASE_URL not found in environment variables")

    def _fetch_category(self, category: str, query: str, page_size: int) -> list[dict]:
        """
        Fetch articles for a single category/query.

        Args:
            category: label identifying which risk category this query covers
            query: the actual search string sent to NewsAPI
            page_size: max articles to fetch for this category

        Returns:
            A list of standardized article dicts, tagged with their category
        """
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",
            # Restrict matching to title + description only, not full body.
            # This avoids false positives where a keyword appears incidentally
            # deep in an unrelated article's body text.
            "searchIn": "title,description",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch news for category '{category}': {e}")
            return []

        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"NewsAPI error for '{category}': {data.get('message')}")
            return []

        articles = data.get("articles", [])

        cleaned_articles = []
        for article in articles:
            cleaned_articles.append({
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
                "description": article.get("description"),
                "published_at": article.get("publishedAt"),
                "url": article.get("url"),
                "category": category,  # tag which risk category this came from
                "fetched_at": datetime.now(UTC).isoformat(),
            })

        return cleaned_articles

    def fetch(self, page_size_per_category: int = 10) -> list[dict]:
        """
        Fetch articles across all disruption risk categories.

        Args:
            page_size_per_category: how many articles to fetch per category
                (default 10 — with 5 categories, that's up to 50 articles total
                per run, staying well within NewsAPI's free tier limits)

        Returns:
            A combined, deduplicated list of articles from all categories
        """
        logger.info(f"Fetching news across {len(self.QUERY_CATEGORIES)} risk categories")

        all_articles = []
        seen_urls = set()  # track URLs to avoid duplicate articles across categories

        for category, query in self.QUERY_CATEGORIES.items():
            articles = self._fetch_category(category, query, page_size_per_category)

            for article in articles:
                url = article.get("url")
                # An article could legitimately match multiple categories
                # (e.g. a strike caused by a trade dispute) — we keep only
                # the first occurrence to avoid duplicate rows downstream
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(article)

        logger.info(f"Fetched {len(all_articles)} unique articles across all categories")
        return all_articles


if __name__ == "__main__":
    collector = NewsCollector()
    articles = collector.fetch()

    print(f"\nFetched {len(articles)} unique articles:\n")
    for article in articles[:15]:
        print(f"[{article['category']}] {article['title']} ({article['source']})")
