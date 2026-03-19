"""Overview page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
from src.config import SUPPORTED_MODELS, LORA_CONFIG

st.title("📊 LoRA Fine-Tuning — Overview")
st.markdown("""
Fine-tune large language models efficiently using **LoRA (Low-Rank Adaptation)**.
This pipeline supports multiple models, configurable LoRA parameters, and evaluation.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Supported Models")
    for name, path in SUPPORTED_MODELS.items():
        st.markdown(f"- **{name}**: `{path.split('/')[-1]}`")

with col2:
    st.subheader("LoRA Config")
    st.json(LORA_CONFIG)

with col3:
    st.subheader("Capabilities")
    st.markdown("- LoRA / QLoRA fine-tuning\n- Chat & instruction formats\n- ROUGE / BLEU evaluation\n- vLLM deployment\n- FastAPI serving")

import numpy as np
models = list(SUPPORTED_MODELS.keys())
param_counts = [7_000_000_000, 7_000_000_000, 3_800_000_000, 67_000_000]
trainable = [f"{p * 0.001:.0f}M" for p in [0.1, 0.1, 0.1, 0.5]]
fig = px.bar(x=models, y=param_counts, labels={"x": "Model", "y": "Parameters"},
             title="Model Sizes", color=param_counts, color_continuous_scale="Blues")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)
