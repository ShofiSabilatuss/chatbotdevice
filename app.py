import streamlit as st
from model import chatbot_response

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Apple Device Support Chatbot",
    page_icon="🍎",
    layout="centered"
)

# =====================
# HIDE STREAMLIT DEFAULT UI
# =====================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================
# TITLE
# =====================
st.markdown(
    "<h3 style='text-align:center;'>🍎 Apple Device Support Chatbot</h3>",
    unsafe_allow_html=True
)

# =====================
# SESSION STATE
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================
# CHAT HISTORY
# =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================
# CHAT INPUT (INI KUNCI)
# =====================
user_input = st.chat_input("Type your message...")

if user_input:
    # user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # bot message
    bot_reply = chatbot_response(user_input)
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()

# =====================
# RESET BUTTON
# =====================
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()
