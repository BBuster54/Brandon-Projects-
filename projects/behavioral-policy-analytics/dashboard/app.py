"""Interactive dashboard for policy, sentiment, causal, and prediction views."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "reports"

st.set_page_config(page_title="Behavioral Policy Analytics", layout="wide")
st.title("Behavioral Policy Analytics Dashboard")
st.caption("Data source: local master HPI (derived from hpi_master.csv)")

city = st.sidebar.selectbox("City", ["Los Angeles", "New York City"])
city_key = "la" if city == "Los Angeles" else "nyc"
city_dir = REPORTS / city_key

policy_default = "2023-04-01" if city_key == "la" else "2019-06-14"
policy_date = st.sidebar.date_input("Policy Date", value=pd.to_datetime(policy_default).date())

monthly_path = city_dir / "monthly_series.csv"
summary_path = city_dir / "policy_summary.csv"
causal_path = city_dir / "causal_effects.csv"
sent_daily_path = city_dir / "sentiment_daily.csv"
pred_summary_path = city_dir / "lag_prediction_summary.csv"
topic_path = city_dir / "topic_evolution.csv"

if not monthly_path.exists():
    st.warning(f"Missing data for {city}. Run: python src/run_project.py {city_key}-case")
    st.stop()

monthly = pd.read_csv(monthly_path)
monthly["month"] = pd.to_datetime(monthly["month"])

col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["monthly_avg_value"],
            mode="lines+markers",
            name="Housing Index",
        )
    )
    fig.update_layout(title=f"{city} Housing Index", xaxis_title="Month", yaxis_title="Index")
    fig.add_vline(x=pd.to_datetime(policy_date), line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        metric_map = {r["metric"]: r["value"] for _, r in summary.iterrows()}
        st.metric("Pre-policy avg", f"{metric_map.get('pre_policy_avg', float('nan')):.2f}")
        st.metric("Post-policy avg", f"{metric_map.get('post_policy_avg', float('nan')):.2f}")
        st.metric("Percent change", f"{metric_map.get('percent_change', float('nan')):.2f}%")

if sent_daily_path.exists():
    sent = pd.read_csv(sent_daily_path)
    sent["date"] = pd.to_datetime(sent["date"])
    st.subheader("Sentiment vs Discussion Volume")
    fig_sent = go.Figure()
    fig_sent.add_trace(
        go.Scatter(
            x=sent["date"],
            y=sent["avg_compound"],
            mode="lines+markers",
            name="Average Sentiment",
            yaxis="y1",
        )
    )
    fig_sent.add_trace(
        go.Bar(
            x=sent["date"],
            y=sent["posts"],
            name="Posts",
            opacity=0.25,
            yaxis="y2",
        )
    )
    fig_sent.update_layout(
        title="Average Sentiment",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Avg Sentiment"),
        yaxis2=dict(title="Posts", overlaying="y", side="right"),
        barmode="overlay",
    )
    st.plotly_chart(fig_sent, use_container_width=True)

if causal_path.exists():
    st.subheader("Counterfactual and Treatment Effect")
    causal = pd.read_csv(causal_path)
    causal["month"] = pd.to_datetime(causal["month"])
    fig_causal = go.Figure()
    fig_causal.add_trace(go.Scatter(x=causal["month"], y=causal["y"], mode="lines", name="Observed"))
    fig_causal.add_trace(
        go.Scatter(x=causal["month"], y=causal["counterfactual"], mode="lines", name="Counterfactual")
    )
    fig_causal.update_layout(title="Observed vs Counterfactual", xaxis_title="Month", yaxis_title="Outcome")
    fig_causal.add_vline(x=pd.to_datetime(policy_date), line_dash="dash", line_color="red")
    st.plotly_chart(fig_causal, use_container_width=True)

    st.caption("Scenario simulation: shift policy date and compare pre/post windows instantly.")
    sim = monthly.copy()
    sim["window"] = sim["month"].apply(lambda x: "Post" if x >= pd.to_datetime(policy_date) else "Pre")
    sim_out = sim.groupby("window", as_index=False)["monthly_avg_value"].mean()
    st.dataframe(sim_out, use_container_width=True)

if pred_summary_path.exists():
    st.subheader("Lagged Prediction Summary")
    st.dataframe(pd.read_csv(pred_summary_path), use_container_width=True)

if topic_path.exists():
    st.subheader("Topic Evolution")
    topic = pd.read_csv(topic_path)
    topic["month"] = pd.to_datetime(topic["month"])
    fig_topic = go.Figure()
    for col in topic.columns:
        if col == "month":
            continue
        fig_topic.add_trace(
            go.Scatter(
                x=topic["month"],
                y=topic[col],
                mode="lines",
                stackgroup="topics",
                name=col,
            )
        )
    fig_topic.update_layout(title="Topic Dynamics", xaxis_title="Month", yaxis_title="Weight")
    st.plotly_chart(fig_topic, use_container_width=True)

compare_path = REPORTS / "comparison" / "cross_city_comparison.csv"
if compare_path.exists():
    st.subheader("Cross-city Comparison")
    st.dataframe(pd.read_csv(compare_path), use_container_width=True)
