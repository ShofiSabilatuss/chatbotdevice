import streamlit as st
from model import chatbot_response

st.set_page_config(
    page_title="Apple Device Support Chatbot",
    layout="centered"
)

st.title("🍎 Apple Device Support Chatbot")

# Session state untuk chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat
user_input = st.chat_input("Ketik pertanyaan Anda...")

if user_input:
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Bot response
    bot_reply = chatbot_response(user_input)
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()

# Reset chat
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()
