"""LoRA Fine-Tuning — Main Streamlit App."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(page_title="LoRA Fine-Tuning", layout="wide", page_icon="🔧")

st.markdown("""<style>
    [data-testid="stSidebar"] { background-color: #262730; }
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #1f77b4; }
</style>""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_💬_Chat.py", title="Chat", icon="💬"),
    st.Page("pages/3_📈_Training_Metrics.py", title="Training", icon="📈"),
    st.Page("pages/4_📋_Evaluation.py", title="Evaluation", icon="📋"),
    st.Page("pages/5_⚙️_Configuration.py", title="Configuration", icon="⚙️"),
])
pg.run()
