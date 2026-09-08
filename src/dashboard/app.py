import time
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# ── Page configuration ─────────────────────────────────────────────────────
# Must be the first Streamlit command called
st.set_page_config(
    page_title="Supply Chain Disruption Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API configuration ──────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000/api"
REFRESH_INTERVAL = 300  # 5 minutes in seconds


# ── API helper functions ───────────────────────────────────────────────────

def fetch_current_risk():
    """Fetches current risk assessment from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/risk/current", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch current risk: {e}")
        return None


def fetch_risk_trend(windows: int = 48):
    """Fetches risk trend data from the API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/risk/trend",
            params={"windows": windows},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch risk trend: {e}")
        return None


def fetch_top_articles(limit: int = 15, min_score: float = 0.6):
    """Fetches highest-risk articles from the API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/risk/articles",
            params={"limit": limit, "min_score": min_score},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch articles: {e}")
        return []


# ── Color mapping ──────────────────────────────────────────────────────────

RISK_COLORS = {
    "Low": "#28a745",     # green
    "Medium": "#ffc107",  # amber
    "High": "#dc3545",    # red
}

CATEGORY_COLORS = {
    "geopolitical": "#e74c3c",
    "natural_disaster": "#e67e22",
    "labor_dispute": "#9b59b6",
    "trade_policy": "#3498db",
    "logistics": "#2ecc71",
}


# ── Dashboard layout ───────────────────────────────────────────────────────

def render_header():
    """Renders the dashboard title and last-updated timestamp."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚢 Supply Chain Disruption Predictor")
        st.caption("Real-time risk intelligence powered by NLP and ML")
    with col2:
        st.metric(
            label="Last Updated",
            value=datetime.now().strftime("%H:%M:%S"),
        )


def render_current_risk(data: dict):
    """Renders the current risk level as a prominent metric card."""
    if not data:
        st.warning("No risk data available. Ensure the scheduler is running.")
        return

    risk = data["risk"]
    signals = data["signals"]
    level = risk["level"]
    score = risk["score"]
    color = RISK_COLORS.get(level, "#6c757d")

    st.markdown("---")
    st.subheader("Current Disruption Risk")

    # Main risk metric — large and prominent
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style='text-align:center; padding:20px;
                        background-color:{color}22;
                        border: 2px solid {color};
                        border-radius:10px;'>
                <h1 style='color:{color}; margin:0;'>{level}</h1>
                <h2 style='color:{color}; margin:0;'>{score:.3f}</h2>
                <p style='color:#666; margin:0;'>Disruption Risk Index</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric(
            label="📰 Avg News Risk",
            value=f"{signals['news']['avg_risk']:.3f}",
            help="Average NLP risk score across collected articles"
        )
        st.metric(
            label="🔴 High Risk Articles",
            value=signals["news"]["high_risk_articles"],
            help="Articles scoring above 0.7 risk threshold"
        )

    with col3:
        st.metric(
            label="🌊 Avg Wind Speed",
            value=f"{signals['weather']['avg_wind_speed_ms']:.1f} m/s",
            help="Average wind speed across 5 tracked ports"
        )
        st.metric(
            label="⛈️ Severe Weather Ports",
            value=signals["weather"]["severe_ports"],
            help="Number of ports experiencing severe weather"
        )

    with col4:
        st.metric(
            label="🛢️ Oil Price Change",
            value=f"{signals['market']['avg_oil_change_pct']:+.2f}%",
            delta=signals["market"]["avg_oil_change_pct"],
            delta_color="inverse",  # price up = bad (inverse coloring)
            help="Average Brent + WTI crude price change"
        )
        st.metric(
            label="🚢 Shipping Stock Change",
            value=f"{signals['market']['avg_shipping_stock_change_pct']:+.2f}%",
            delta=signals["market"]["avg_shipping_stock_change_pct"],
            help="Average ZIM + MATX stock price change"
        )

    st.caption(
        f"Window: {data['window_start'][:19]} → {data['window_end'][:19]} UTC"
    )
    st.markdown(f"*{risk['description']}*")


def render_trend_chart(trend_data: dict):
    """Renders an interactive line chart of risk over time."""
    if not trend_data or not trend_data.get("windows"):
        st.info("Insufficient trend data. More data accumulates every 30 minutes.")
        return

    st.markdown("---")
    st.subheader(
        f"Risk Trend — {trend_data['trend'].capitalize()} "
        f"(avg: {trend_data['avg_risk']:.3f})"
    )

    df = pd.DataFrame(trend_data["windows"])
    df["window_end"] = pd.to_datetime(df["window_end"], format="mixed", utc=True)
    df = df.sort_values("window_end")

    fig = go.Figure()

    # Main risk line
    fig.add_trace(go.Scatter(
        x=df["window_end"],
        y=df["risk_score"],
        mode="lines+markers",
        name="Disruption Risk Index",
        line=dict(color="#dc3545", width=2),
        marker=dict(size=6),
        hovertemplate="<b>Risk: %{y:.3f}</b><br>%{x}<extra></extra>",
    ))

    # News risk line (secondary signal)
    fig.add_trace(go.Scatter(
        x=df["window_end"],
        y=df["news_risk"],
        mode="lines",
        name="News Risk",
        line=dict(color="#3498db", width=1, dash="dot"),
        hovertemplate="News Risk: %{y:.3f}<br>%{x}<extra></extra>",
    ))

    # Risk level threshold lines
    fig.add_hline(y=0.6, line_dash="dash", line_color="#dc3545",
                  annotation_text="High Risk", annotation_position="right")
    fig.add_hline(y=0.3, line_dash="dash", line_color="#ffc107",
                  annotation_text="Medium Risk", annotation_position="right")

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=80, t=40, b=0),
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_articles_table(articles: list):
    """Renders high-risk articles as an interactive table."""
    if not articles:
        st.info("No high-risk articles found above the current threshold.")
        return

    st.markdown("---")
    st.subheader(f"Top Risk Articles ({len(articles)} shown)")

    df = pd.DataFrame(articles)
    df = df[["risk_score", "category", "sentiment_label", "title", "source"]]
    df["risk_score"] = df["risk_score"].round(3)
    df.columns = ["Risk Score", "Category", "Sentiment", "Title", "Source"]

    # Color-code by risk score
    def color_risk(val):
        if val >= 0.7:
            return "background-color: #dc354522"
        elif val >= 0.4:
            return "background-color: #ffc10722"
        return "background-color: #28a74522"

    styled = df.style.applymap(color_risk, subset=["Risk Score"])
    st.dataframe(styled, use_container_width=True, height=400)


def render_sidebar():
    """Renders sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Settings")

        min_score = st.slider(
            "Minimum article risk score",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Filter articles below this risk score"
        )

        article_limit = st.slider(
            "Max articles to show",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
        )

        trend_windows = st.slider(
            "Trend history (windows)",
            min_value=4,
            max_value=96,
            value=24,
            step=4,
            help="Each window = 30 minutes"
        )

        st.markdown("---")
        st.caption(f"API: {API_BASE_URL}")
        st.caption("Refreshes every 5 minutes")

        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    return min_score, article_limit, trend_windows


# ── Main app ───────────────────────────────────────────────────────────────

def main():
    render_header()

    # Render sidebar and get user settings
    min_score, article_limit, trend_windows = render_sidebar()

    # Fetch all data
    with st.spinner("Fetching latest risk data..."):
        current_risk = fetch_current_risk()
        trend_data = fetch_risk_trend(windows=trend_windows)
        articles = fetch_top_articles(
            limit=article_limit,
            min_score=min_score
        )

    # Render sections
    render_current_risk(current_risk)
    render_trend_chart(trend_data)
    render_articles_table(articles)

    # Auto-refresh every 5 minutes
    # st.rerun() reloads the whole page — combined with time.sleep()
    # this creates a simple polling loop
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
