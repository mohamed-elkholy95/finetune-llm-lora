"""Training metrics page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("📈 Training Metrics")

st.markdown("Loss curves and training history for fine-tuned models.")

# Mock training data
epochs = list(range(1, 4))
loss = [2.45, 1.82, 1.35]
eval_loss = [2.60, 1.95, 1.48]
learning_rate = [2e-4, 1.8e-4, 1.2e-4]

fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=loss, name="Train Loss", mode="lines+markers",
                         line=dict(color="#1f77b4", width=2)))
fig.add_trace(go.Scatter(x=epochs, y=eval_loss, name="Eval Loss", mode="lines+markers",
                         line=dict(color="#ff7f0e", width=2, dash="dash")))
fig.update_layout(title="Training & Evaluation Loss", xaxis_title="Epoch", yaxis_title="Loss",
                  paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=epochs, y=learning_rate, name="Learning Rate",
                          mode="lines+markers", line=dict(color="#2ca02c", width=2)))
fig2.update_layout(title="Learning Rate Schedule", xaxis_title="Epoch", yaxis_title="LR",
                   paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("Epochs", "3/3")
    st.metric("Final Train Loss", "1.35")
with col2:
    st.metric("Best Eval Loss", "1.48")
    st.metric("Trainable Params", "0.1%")
