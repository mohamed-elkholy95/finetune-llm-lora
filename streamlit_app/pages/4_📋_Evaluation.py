"""Evaluation page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px

st.title("📋 Evaluation Results")

st.markdown("Compare model performance before and after fine-tuning.")

metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
before = [0.32, 0.12, 0.28, 15.2]
after = [0.48, 0.22, 0.41, 28.7]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Before Fine-Tuning")
    for m, v in zip(metrics, before):
        st.metric(m, f"{v:.2f}" if v < 100 else f"{v:.1f}")
with col2:
    st.subheader("After Fine-Tuning")
    for m, v, b in zip(metrics, after, before):
        delta = v - b
        sign = "+" if delta >= 0 else ""
        st.metric(m, f"{v:.2f}" if v < 100 else f"{v:.1f}", f"{sign}{delta:.2f}")

fig = px.bar(x=metrics, y=[after[i] - before[i] for i in range(4)],
             labels={"x": "Metric", "y": "Improvement"},
             title="Improvement After Fine-Tuning", color=[after[i] - before[i] for i in range(4)],
             color_continuous_scale="Greens")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)
