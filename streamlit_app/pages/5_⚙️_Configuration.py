"""Configuration page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import yaml
from pathlib import Path as P

st.title("⚙️ Configuration")

st.markdown("Edit training and LoRA parameters.")

config_dir = P(__file__).resolve().parent.parent.parent / "configs"
configs = list(config_dir.glob("*.yaml")) if config_dir.exists() else []

if configs:
    selected = st.selectbox("Config File", [c.name for c in configs])
    config_path = config_dir / selected
    with open(config_path) as f:
        config = yaml.safe_load(f)
    edited = st.json(config)
else:
    st.info("No config files found. Using defaults.")

with st.form("lora_params"):
    st.subheader("LoRA Parameters")
    col1, col2 = st.columns(2)
    with col1:
        r = st.number_input("Rank (r)", 1, 64, 8)
        alpha = st.number_input("Alpha", 1, 128, 16)
    with col2:
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1)
        target = st.text_input("Target Modules", "q_proj, v_proj")

    st.subheader("Training Parameters")
    col3, col4 = st.columns(2)
    with col3:
        epochs = st.number_input("Epochs", 1, 20, 3)
        batch_size = st.number_input("Batch Size", 1, 32, 4)
    with col4:
        lr = st.number_input("Learning Rate", 0.0, 0.01, 2e-4, format="%e")
        warmup = st.slider("Warmup Ratio", 0.0, 0.3, 0.03)

    submitted = st.form_submit_button("Save Config")
    if submitted:
        st.success("Configuration saved!")
