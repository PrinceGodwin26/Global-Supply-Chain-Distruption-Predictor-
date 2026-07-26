from datetime import datetime, UTC
from transformers import pipeline
from loguru import logger

from src.utils.database import get_connection
from src.processing.text_cleaner import build_input_text


# Keywords that strongly indicate supply-chain relevance.
# An article must contain at least one of these in its title or description
# to be considered relevant enough for full NLP scoring.
# These are deliberately specific multi-word phrases or unambiguous single
# words — "port" alone could mean "wine port" or "porting code", but
# "port closure" is unambiguous.
SUPPLY_CHAIN_KEYWORDS = [
    # Highly specific — unambiguous supply chain terms
    "supply chain", "port closure", "port congestion", "shipping delay",
    "shipping route", "trade route", "container shortage", "freight rate",
    "cargo shipment", "port workers", "dock workers", "factory shutdown",
    "trade war", "trade ban", "export ban", "import ban",

    # Specific enough in context
    "global shipping", "global trade", "ocean freight", "air freight",
    "vessel", "tanker", "strait", "suez", "hormuz", "malacca",
    "supply disruption", "logistics disruption", "customs delay",

    # Single words kept only because they're genuinely domain-specific
    "tariff", "freight", "logistics",

    # Trade volume phrases — specific enough to avoid false matches
    "export growth", "export target", "import growth", "trade surplus",
    "trade deficit", "trade volume", "bilateral trade",
]

# Disruption-signal keywords — words that indicate something is actually
# going wrong, not just being discussed. An article scoring above 0.9
# but missing ALL of these is likely a false positive (e.g. a product
# announcement using supply-chain vocabulary without any actual disruption).
DISRUPTION_SIGNALS = [
    # Specific disruption events — unambiguous
    "closure", "closed", "shutdown", "shut down", "disrupted",
    "delayed", "shortage", "blocked", "blockade", "strike",
    "suspended", "halted", "congestion", "bottleneck", "backlog",
    "evacuation", "damaged", "destroyed", "flood", "earthquake",
    "typhoon", "hurricane", "wildfire", "seized", "attacked",
    "collision", "explosion",

    # Kept as phrases only (not standalone words) to avoid false matches
    "supply disruption", "trade disruption", "shipping disruption",
    "port disruption", "major delay", "significant delay",
    "trade conflict", "armed conflict", "military conflict",
    "export ban", "import ban", "trade sanctions",
]

# Escalation signals — words that indicate cost/price/volume increases
# which are disruption indicators even when framed positively.
# e.g. "Shipping costs soar" scores POSITIVE sentiment but IS a disruption.
# When these appear with supply-chain keywords, we treat them as risk signals.
ESCALATION_SIGNALS = [
    "soar", "soaring", "surge", "surging", "spike", "spiking",
    "skyrocket", "skyrocketing", "jump", "jumped", "climb", "climbing",
    "stack up", "stacking up", "pile up", "piling up", "backlog",
    "overwhelm", "overwhelmed", "capacity", "bottleneck",
]

# Category weights reflect how precisely each category's queries
# target supply-chain content. Higher weight = more trust in the
# raw sentiment score from that category.
CATEGORY_WEIGHTS = {
    "logistics": 1.0,
    "trade_policy": 0.9,
    "labor_dispute": 0.85,
    "natural_disaster": 0.75,
    "geopolitical": 0.75,
}


def is_supply_chain_relevant(title: str | None, description: str | None) -> bool:
    """
    Checks whether an article is likely supply-chain relevant by looking
    for domain-specific keywords in the title or description.

    This acts as a pre-filter before calling the expensive NLP model —
    filtering out articles that mention a flood in a human interest story
    (no supply chain keywords) vs a flood that closes a major port
    (contains "port closure" or "logistics").

    Args:
        title: article headline
        description: article summary

    Returns:
        True if at least one supply-chain keyword is found, False otherwise
    """
    combined = f"{title or ''} {description or ''}".lower()
    return any(keyword in combined for keyword in SUPPLY_CHAIN_KEYWORDS)


class NLPProcessor:
    """
    Scores news articles for supply chain disruption risk using a
    pre-trained DistilBERT sentiment classifier, combined with a
    relevance pre-filter and category-based weighting.

    Three-step scoring pipeline:
    1. Relevance check  — filter out non-supply-chain articles early
    2. Sentiment score  — DistilBERT negative sentiment = disruption risk
    3. Category weight  — adjust score based on query precision per category

    Risk score interpretation:
        0.0 - 0.3: Low risk (positive/neutral or irrelevant news)
        0.3 - 0.6: Medium risk (ambiguous or mixed signals)
        0.6 - 1.0: High risk (clear disruption indicators)

    Known limitations (to be addressed in future fine-tuning phase):
        1. Positively-framed disruption articles (e.g. "freight modernization")
           score near 0 because the model reads tone, not subject matter.
        2. General business/stock articles using supply-chain vocabulary
           (e.g. "free shipping", "logistics software") can pass the keyword
           filter and score medium-high despite being irrelevant.
        3. These limitations are inherent to using a general sentiment model
           without domain-specific fine-tuning on labeled supply chain data.
           Resolution: collect labeled data and fine-tune in a later phase.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        logger.info(f"Loading NLP model: {self.MODEL_NAME}")
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            device=-1  # force CPU
        )
        logger.info("NLP model loaded successfully")

    def score_article(self, title: str | None, description: str | None, category: str) -> dict:
        """
        Scores a single article using the three-step pipeline.

        Args:
            title: article headline
            description: article summary
            category: which query category this article came from

        Returns:
            A dict with 'risk_score' (float 0-1), 'sentiment_label' (str),
            and 'relevant' (bool) indicating whether it passed the relevance filter
        """
        # Step 1: Relevance filter
        if not is_supply_chain_relevant(title, description):
            return {
                "risk_score": 0.1,
                "sentiment_label": "IRRELEVANT",
                "relevant": False,
            }

        # Step 2: Sentiment scoring
        input_text = build_input_text(title, description)

        if not input_text:
            return {"risk_score": 0.1, "sentiment_label": "NEUTRAL", "relevant": False}

        result = self.classifier(input_text)[0]
        label = result["label"]
        confidence = result["score"]

        raw_risk = confidence if label == "NEGATIVE" else 1 - confidence

        # Escalation signal override:
        # If the model scores POSITIVE but the article contains escalation
        # signals (costs soaring, ships stacking up, backlogs growing),
        # these are disruption indicators despite positive framing.
        # Override: treat as medium-high risk (0.7) rather than near-zero.
        if label == "POSITIVE" and confidence > 0.7:
            combined_text = f"{title or ''} {description or ''}".lower()
            has_escalation = any(
                signal in combined_text for signal in ESCALATION_SIGNALS
            )
            if has_escalation:
                raw_risk = 0.7
                label = "ESCALATION"
                logger.debug(
                    f"Escalation override applied: {str(title)[:60]}"
                )

        # Step 3: Apply category weight
        weight = CATEGORY_WEIGHTS.get(category, 0.8)
        weighted_risk = raw_risk * weight

        # Clamp to [0, 1] — floating point math can occasionally produce
        # values just outside this range
        final_risk = max(0.0, min(1.0, weighted_risk))

        # Secondary disruption-signal check:
        # If the score is high but no disruption-signal word appears anywhere
        # in the article text, cap the score at 0.75.
        # Rationale: genuine disruption articles almost always contain at least
        # one explicit disruption word. High-scoring false positives (product
        # announcements, stock articles using supply-chain vocabulary) typically
        # don't. This catches that class of error without touching true positives.
        if final_risk > 0.9:
            combined_text = f"{title or ''} {description or ''}".lower()
            has_disruption_signal = any(
                signal in combined_text for signal in DISRUPTION_SIGNALS
            )
            if not has_disruption_signal:
                final_risk = 0.75
                logger.debug(
                    f"Score capped at 0.75 (no disruption signal found): "
                    f"{str(title)[:60]}"
                )

        return {
            "risk_score": round(final_risk, 4),
            "sentiment_label": label,
            "relevant": True,
        }

    def process_unscored_articles(self) -> int:
        """
        Fetches all unscored articles from the database, scores them,
        and saves results back. Safe to run multiple times — only picks
        up articles where risk_score IS NULL.

        Returns:
            The number of articles successfully scored
        """
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, title, description, category
            FROM news_articles
            WHERE risk_score IS NULL
            ORDER BY published_at DESC
        """)

        articles = cursor.fetchall()
        logger.info(f"Found {len(articles)} unscored articles to process")

        if not articles:
            cursor.close()
            conn.close()
            return 0

        scored_count = 0
        relevant_count = 0

        for article_id, title, description, category in articles:
            try:
                scores = self.score_article(title, description, category)

                cursor.execute("""
                    UPDATE news_articles
                    SET risk_score = %s,
                        sentiment_label = %s,
                        processed_at = %s
                    WHERE id = %s
                """, (
                    scores["risk_score"],
                    scores["sentiment_label"],
                    datetime.now(UTC),
                    article_id,
                ))

                scored_count += 1
                if scores["relevant"]:
                    relevant_count += 1

            except Exception as e:
                logger.error(f"Failed to score article {article_id}: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Scored {scored_count} articles — {relevant_count} relevant, {scored_count - relevant_count} filtered as irrelevant")
        return scored_count


if __name__ == "__main__":
    processor = NLPProcessor()
    count = processor.process_unscored_articles()

    if count > 0:
        conn = get_connection()
        cursor = conn.cursor()

        # Show top 10 highest-risk relevant articles
        cursor.execute("""
            SELECT title, category, risk_score, sentiment_label
            FROM news_articles
            WHERE risk_score IS NOT NULL
            AND sentiment_label != 'IRRELEVANT'
            ORDER BY risk_score DESC
            LIMIT 10
        """)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        print(f"\nTop 10 highest-risk articles:\n")
        for title, category, risk_score, label in results:
            print(f"[{risk_score:.3f}] [{category}] {title[:70]}")
