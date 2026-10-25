"""PriceRadar - Streamlit Dashboard."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.config_loader import PROJECT_ROOT

st.set_page_config(
    page_title="PriceRadar",
    page_icon="📊",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #e94560;
    }
    .insight-box h4 {
        color: #e94560;
        margin: 0 0 8px 0;
    }
    .insight-box p {
        color: #ccc;
        margin: 0;
        font-size: 0.95rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px 20px;
        border: 1px solid #1a1a4e;
    }
    div[data-testid="stMetric"] label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data
def load_data(filename: str, subdir: str = "anomalies") -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / subdir / filename
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else None)
    return pd.DataFrame()

@st.cache_data
def load_forecast(filename: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "forecasts" / filename
    if path.exists():
        return pd.read_csv(path, parse_dates=["ds"] if "ds" in pd.read_csv(path, nrows=0).columns else None)
    return pd.DataFrame()

@st.cache_data
def load_news() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "news_cleaned.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


# --- Sidebar ---
st.sidebar.markdown("## 📊 PriceRadar")
st.sidebar.caption("Track commodity prices, spot unusual changes, and see future price predictions.")

page = st.sidebar.radio(
    "📌 Go to",
    ["🏠 Home", "🔍 Unusual Price Changes", "📈 Price Predictions", "📰 Latest News", "📋 Data Health"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Choose Data Source**")
source = st.sidebar.selectbox(
    "source",
    ["Yahoo Finance (Daily Prices)", "FRED (Monthly Agriculture)"],
    label_visibility="collapsed",
)

if "Yahoo" in source:
    anomaly_df = load_data("yahoo_anomalies.csv")
    prophet_df = load_forecast("prophet_yahoo.csv")
    lgbm_df = load_forecast("lgbm_yahoo.csv")
    price_col = "close"
    source_label = "Yahoo Finance"
else:
    anomaly_df = load_data("fred_anomalies.csv")
    prophet_df = load_forecast("prophet_fred.csv")
    lgbm_df = load_forecast("lgbm_fred.csv")
    price_col = "price"
    source_label = "FRED"

news_df = load_news()
commodities = anomaly_df["commodity"].unique().tolist() if not anomaly_df.empty else []

# Color map for commodities
COLORS = {
    "Crude Oil WTI": "#e94560",
    "Gold": "#f5a623",
    "Natural Gas": "#4fc3f7",
    "Wheat": "#8bc34a",
    "Rice": "#ff9800",
    "Sugar": "#ce93d8",
}


def get_color(commodity: str) -> str:
    return COLORS.get(commodity, "#4fc3f7")


# === PAGE: Home ===
if page == "🏠 Home":
    st.markdown('<p class="main-header">📊 PriceRadar Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monitor commodity prices, detect unusual movements, and forecast future trends — all in one place.</p>', unsafe_allow_html=True)

    if anomaly_df.empty:
        st.warning("⚠️ No data found. Please run the data pipeline first.")
        st.stop()

    # KPI Cards
    total_anomalies = int(anomaly_df["anomaly_combined"].sum())
    high_sev = int((anomaly_df["anomaly_severity"] == "high").sum())
    date_range = f"{anomaly_df['date'].min().strftime('%b %Y')} — {anomaly_df['date'].max().strftime('%b %Y')}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Time Period", date_range)
    with col2:
        st.metric("📦 Commodities Tracked", len(commodities))
    with col3:
        st.metric("⚠️ Unusual Price Events", f"{total_anomalies}")
    with col4:
        st.metric("🔴 Major Events", f"{high_sev}")

    st.markdown("---")

    # Price charts
    st.subheader("💰 Price History")
    st.caption("Each dot marks an unusual price movement. Red = major, Orange = moderate.")

    for commodity in commodities:
        cdf = anomaly_df[anomaly_df["commodity"] == commodity].sort_values("date")
        anomalies = cdf[cdf["anomaly_combined"] == 1]
        color = get_color(commodity)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cdf["date"], y=cdf[price_col],
            mode="lines", name="Price",
            line=dict(width=1.5, color=color),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>",
        ))

        if not anomalies.empty:
            high = anomalies[anomalies["anomaly_severity"] == "high"]
            medium = anomalies[anomalies["anomaly_severity"] == "medium"]

            if not medium.empty:
                fig.add_trace(go.Scatter(
                    x=medium["date"], y=medium[price_col],
                    mode="markers", name="Moderate Change",
                    marker=dict(color="#f5a623", size=7, line=dict(width=1, color="white")),
                    hovertemplate="⚠️ Moderate<br>Date: %{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>",
                ))
            if not high.empty:
                fig.add_trace(go.Scatter(
                    x=high["date"], y=high[price_col],
                    mode="markers", name="Major Change",
                    marker=dict(color="#e94560", size=9, symbol="diamond",
                                line=dict(width=1, color="white")),
                    hovertemplate="🔴 Major<br>Date: %{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>",
                ))

        fig.update_layout(
            title=dict(text=commodity, font=dict(size=16)),
            xaxis_title="", yaxis_title="Price (USD)",
            height=320,
            margin=dict(l=50, r=20, t=40, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.subheader("📊 Summary")
    summary = anomaly_df.groupby("commodity").agg(
        data_points=("date", "count"),
        unusual_events=("anomaly_combined", "sum"),
        major_events=("anomaly_severity", lambda x: (x == "high").sum()),
    ).reset_index()
    summary.columns = ["Commodity", "Total Days", "Unusual Events", "Major Events"]
    summary["Event Rate"] = (summary["Unusual Events"] / summary["Total Days"] * 100).round(1).astype(str) + "%"
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Key insight box
    worst = summary.loc[summary["Unusual Events"].idxmax()]
    st.markdown(f"""
    <div class="insight-box">
        <h4>💡 Key Insight</h4>
        <p><strong>{worst['Commodity']}</strong> had the most unusual price movements
        ({int(worst['Unusual Events'])} events out of {int(worst['Total Days'])} trading days),
        suggesting it was the most volatile commodity in this period.</p>
    </div>
    """, unsafe_allow_html=True)


# === PAGE: Unusual Price Changes ===
elif page == "🔍 Unusual Price Changes":
    st.markdown('<p class="main-header">🔍 Unusual Price Changes</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Days when prices moved far outside their normal range. The system uses two methods to catch these.</p>', unsafe_allow_html=True)

    if anomaly_df.empty:
        st.warning("No data available.")
        st.stop()

    selected = st.selectbox("Select a commodity", commodities)
    cdf = anomaly_df[anomaly_df["commodity"] == selected].sort_values("date")
    color = get_color(selected)

    # How far from normal chart
    st.subheader("📏 How Far From Normal?")
    st.caption("This chart shows how much the price deviated from its recent average. Values beyond the red lines are flagged as unusual.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["zscore"],
        mode="lines", name="Deviation Score",
        line=dict(width=1, color=color),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Score: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=2.5, line_dash="dash", line_color="#e94560",
                  annotation_text="Upper threshold")
    fig.add_hline(y=-2.5, line_dash="dash", line_color="#e94560",
                  annotation_text="Lower threshold")
    fig.add_hrect(y0=-2.5, y1=2.5, fillcolor="rgba(76,175,80,0.05)", line_width=0,
                  annotation_text="Normal range", annotation_position="top left")
    fig.update_layout(
        height=300, margin=dict(l=50, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Deviation from Average",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # Pattern-based detection
    if "iforest_score" in cdf.columns:
        st.subheader("🧩 Pattern-Based Detection")
        st.caption("An AI model looks at multiple price patterns together to find days that don't fit the usual behavior.")

        fig2 = go.Figure()
        normal = cdf[cdf["anomaly_iforest"] == 0]
        anomaly = cdf[cdf["anomaly_iforest"] == 1]

        fig2.add_trace(go.Scatter(
            x=normal["date"], y=normal["iforest_score"],
            mode="lines", name="Normal",
            line=dict(width=1, color="#555"),
        ))
        fig2.add_trace(go.Scatter(
            x=anomaly["date"], y=anomaly["iforest_score"],
            mode="markers", name="Unusual",
            marker=dict(color="#e94560", size=6, line=dict(width=1, color="white")),
        ))
        fig2.update_layout(
            height=280, margin=dict(l=50, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Anomaly Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig2, use_container_width=True)

    # Event list
    st.subheader("📋 All Detected Events")
    anomaly_rows = cdf[cdf["anomaly_combined"] == 1][["date", price_col, "zscore", "anomaly_severity"]].copy()
    anomaly_rows.columns = ["Date", "Price (USD)", "Deviation Score", "Severity"]
    anomaly_rows["Severity"] = anomaly_rows["Severity"].map({"high": "🔴 Major", "medium": "🟡 Moderate"})
    anomaly_rows = anomaly_rows.sort_values("Date", ascending=False)
    anomaly_rows["Price (USD)"] = anomaly_rows["Price (USD)"].round(2)
    anomaly_rows["Deviation Score"] = anomaly_rows["Deviation Score"].round(2)
    st.dataframe(anomaly_rows, use_container_width=True, hide_index=True)

    st.caption(f"Total: {len(anomaly_rows)} unusual events detected for {selected}")


# === PAGE: Forecasting ===
elif page == "📈 Price Predictions":
    st.markdown('<p class="main-header">📈 Price Predictions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">What our models predict prices will look like. Compare actual prices (blue) vs predictions (red dashed).</p>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model_choice = st.radio("Prediction Model", ["Prophet", "LightGBM"], horizontal=True,
                                help="Prophet: time-series model by Meta. LightGBM: machine learning model by Microsoft.")
    forecast_df = prophet_df if model_choice == "Prophet" else lgbm_df

    if forecast_df.empty:
        st.warning(f"No predictions available for {model_choice}.")
        st.stop()

    forecast_commodities = forecast_df["commodity"].unique().tolist()
    with col_m2:
        selected = st.selectbox("Commodity", forecast_commodities)

    cdf = forecast_df[forecast_df["commodity"] == selected].sort_values("ds")

    fig = go.Figure()

    actual = cdf[cdf["y"].notna()]
    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual["ds"], y=actual["y"],
            mode="lines", name="Actual Price",
            line=dict(color="#4fc3f7", width=2),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Actual: $%{y:.2f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=cdf["ds"], y=cdf["yhat"],
        mode="lines", name="Predicted Price",
        line=dict(color="#e94560", width=2, dash="dash"),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Predicted: $%{y:.2f}<extra></extra>",
    ))

    if "yhat_lower" in cdf.columns and model_choice == "Prophet":
        fig.add_trace(go.Scatter(
            x=pd.concat([cdf["ds"], cdf["ds"][::-1]]),
            y=pd.concat([cdf["yhat_upper"], cdf["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(233, 69, 96, 0.08)",
            line=dict(color="rgba(233,69,96,0)"),
            name="Prediction Range",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(text=f"{selected} — {model_choice} Prediction", font=dict(size=16)),
        xaxis_title="", yaxis_title="Price (USD)",
        height=450,
        margin=dict(l=50, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy
    if not actual.empty:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
        merged = cdf[cdf["y"].notna()].dropna(subset=["yhat"])
        if not merged.empty:
            mae = mean_absolute_error(merged["y"], merged["yhat"])
            mape = mean_absolute_percentage_error(merged["y"], merged["yhat"]) * 100
            accuracy = max(0, 100 - mape)

            st.markdown("### 🎯 How Accurate?")
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Accuracy", f"{accuracy:.1f}%",
                       help="How close predictions are to actual prices. Higher is better.")
            ac2.metric("Average Error", f"${mae:.2f}",
                       help="On average, predictions were off by this much.")
            ac3.metric("Error Rate", f"{mape:.1f}%",
                       help="Average percentage difference between predicted and actual prices.")

            if accuracy >= 90:
                st.success("✅ Model is performing well — predictions are very close to actual prices.")
            elif accuracy >= 75:
                st.info("ℹ️ Model has reasonable accuracy — predictions follow the general trend.")
            else:
                st.warning("⚠️ Model accuracy is limited — commodity prices are inherently hard to predict.")


# === PAGE: News ===
elif page == "📰 Latest News":
    st.markdown('<p class="main-header">📰 Commodity News</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Recent news headlines related to commodity markets. Helps explain why prices moved.</p>', unsafe_allow_html=True)

    if news_df.empty:
        st.warning("No news data available.")
        st.stop()

    # Keyword filter
    all_queries = news_df["query"].unique().tolist()
    query_labels = {q: q.replace("commodity", "").replace("price", "").strip().title() for q in all_queries}

    selected_queries = st.multiselect(
        "🔎 Filter by topic",
        all_queries,
        default=all_queries[:3],
        format_func=lambda x: query_labels.get(x, x),
    )

    filtered = news_df[news_df["query"].isin(selected_queries)].sort_values("date", ascending=False)

    # Stats
    col1, col2 = st.columns(2)
    col1.metric("📰 Total Articles", f"{len(filtered):,}")
    col2.metric("📅 Sources", filtered["source"].nunique() if "source" in filtered.columns else "N/A")

    # Timeline
    st.subheader("📅 News Volume Over Time")
    daily = filtered.groupby(filtered["date"].dt.date).size().reset_index(name="Articles")
    daily.columns = ["Date", "Articles"]
    fig = px.bar(daily, x="Date", y="Articles",
                 color_discrete_sequence=["#4fc3f7"])
    fig.update_layout(
        height=250, margin=dict(l=50, r=20, t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis_title="Number of Articles",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # Headlines
    st.subheader("📋 Recent Headlines")
    display = filtered[["date", "title", "source", "query"]].head(50).copy()
    display.columns = ["Date", "Headline", "Source", "Topic"]
    display["Date"] = display["Date"].dt.strftime("%b %d, %Y")
    display["Topic"] = display["Topic"].map(query_labels)
    st.dataframe(display, use_container_width=True, hide_index=True)


# === PAGE: Data Health ===
elif page == "📋 Data Health":
    st.markdown('<p class="main-header">📋 Data Health Check</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Is our data complete and reliable? This page shows data quality reports and basic stats.</p>', unsafe_allow_html=True)

    # Quick stats
    st.subheader("📊 Dataset Overview")

    stats = []
    for name, subdir, fname in [
        ("Yahoo Daily Prices (Raw)", "raw", "yahoo_commodities.csv"),
        ("FRED Monthly Prices (Raw)", "raw", "fred_commodities.csv"),
        ("News Headlines (Raw)", "raw", "news_headlines.csv"),
        ("Yahoo Prices (Cleaned)", "processed", "yahoo_cleaned.csv"),
        ("FRED Prices (Cleaned)", "processed", "fred_cleaned.csv"),
        ("News Headlines (Cleaned)", "processed", "news_cleaned.csv"),
    ]:
        path = PROJECT_ROOT / "data" / subdir / fname
        if path.exists():
            df = pd.read_csv(path)
            missing = int(df.isnull().sum().sum())
            stats.append({
                "Dataset": name,
                "Rows": f"{len(df):,}",
                "Columns": df.shape[1],
                "Missing Values": missing,
                "Status": "✅ Clean" if missing == 0 else f"⚠️ {missing} missing",
            })

    if stats:
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
    else:
        st.info("No datasets found. Run the pipeline first.")

    # Evidently reports
    st.markdown("---")
    st.subheader("📑 Detailed Quality Reports")
    st.caption("These reports are generated by Evidently AI and show in-depth data quality metrics.")

    reports_dir = PROJECT_ROOT / "reports" / "monitoring"

    if reports_dir.exists():
        report_files = sorted(reports_dir.glob("*.html"))
        if report_files:
            friendly_names = {f.stem: f.stem.replace("_", " ").title() for f in report_files}

            selected_report = st.selectbox(
                "Select a report to view",
                report_files,
                format_func=lambda x: friendly_names.get(x.stem, x.stem),
            )

            if selected_report:
                with open(selected_report, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=700, scrolling=True)
        else:
            st.info("No reports generated yet. Run `python run_mlops.py --step monitor`.")
    else:
        st.info("Reports folder not found. Run the MLOps pipeline first.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {source_label}")
st.sidebar.caption("Built with Streamlit + Plotly")