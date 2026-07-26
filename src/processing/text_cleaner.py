import re


def clean_text(text: str | None) -> str:
    """
    Cleans raw article text before feeding it to the NLP model.

    Raw news text often contains HTML tags, special characters, and
    inconsistent whitespace — all of which add noise to the model's
    input without adding meaning. Cleaning standardizes the input so
    the model focuses on actual words and context.

    Args:
        text: raw text string, possibly None (some articles have no description)

    Returns:
        A cleaned, normalized string safe to pass to the transformer model
    """
    if not text:
        return ""

    # Remove HTML tags — some news sources include markup in their descriptions
    # e.g. "<b>Port closure</b> disrupts shipping" → "Port closure disrupts shipping"
    text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs — they add no semantic value for sentiment analysis
    # e.g. "Read more at https://example.com/article" → "Read more at"
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters except basic punctuation
    # Keeps letters, digits, spaces, and . ! ? , — enough for the model
    # to understand sentence structure and context
    text = re.sub(r"[^\w\s.,!?-]", "", text)

    # Collapse multiple spaces/newlines into a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def build_input_text(title: str | None, description: str | None) -> str:
    """
    Combines title and description into a single input string for the model.

    We use both fields because:
    - The title alone is often too short for context (e.g. just "Port Strike")
    - The description adds crucial context (e.g. "...affecting 300 vessels")
    - Together they give the model the richest possible signal

    Args:
        title: article headline
        description: article summary/description

    Returns:
        A single cleaned string: "title. description" (or just title/description
        if one is missing), truncated to 512 characters — DistilBERT's max input
    """
    title_clean = clean_text(title)
    desc_clean = clean_text(description)

    if title_clean and desc_clean:
        combined = f"{title_clean}. {desc_clean}"
    elif title_clean:
        combined = title_clean
    else:
        combined = desc_clean

    # DistilBERT has a hard limit of 512 tokens — feeding longer text
    # causes it to silently truncate, potentially losing the most important
    # part of the text. We truncate at 512 characters as a safe approximation
    # (characters ≠ tokens, but 512 chars is well within 512 tokens for English)
    return combined[:512]
