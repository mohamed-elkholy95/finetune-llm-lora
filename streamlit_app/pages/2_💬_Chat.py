"""Chat page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from src.inference import mock_generate

st.title("💬 Chat with Fine-Tuned Model")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = mock_generate(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear Chat"):
    st.session_state.messages = []
