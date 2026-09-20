import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Mr. Dhilleswar Local LLaMA Chat Bot", page_icon="🦙")

st.title("🦙 Mr. Dhilleswar LLaMA 3.2 Chat BOT")
st.caption("Powered by Docker Model Runner — running fully on your machine")

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="docker"
)

MODEL = "ai/llama3.2:1B-Q4_0"

# Keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Answer the question in a couple sentences."}
    ]

# Show past messages (skip the system prompt)
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box
user_input = st.chat_input("Ask something about LLaMA 3.2...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            completion = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages
            )
            reply = completion.choices[0].message.content
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})