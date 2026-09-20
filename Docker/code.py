import streamlit as st
from openai import OpenAI


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mr. Dhilleswar LLaMA",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.main-title {
    text-align: center;
    color: #ffffff;
    font-size: 42px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 35px;
}

.welcome-box {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 45px 30px;
    margin: 35px auto;
    max-width: 850px;
    text-align: center;
}

.welcome-icon {
    font-size: 65px;
    margin-bottom: 15px;
}

.welcome-title {
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 18px;
}

.welcome-text {
    color: #9ca3af;
    font-size: 16px;
    line-height: 1.7;
    margin: 8px 0;
}

.sidebar-brand {
    text-align: center;
    color: #ffffff;
    font-size: 24px;
    font-weight: 700;
}

.sidebar-description {
    text-align: center;
    color: #9ca3af;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 50px;
    margin-bottom: 20px;
    padding: 20px;
}

.footer-main {
    color: #9ca3af;
    font-weight: 500;
}

.footer-local {
    color: #6b7280;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DOCKER MODEL RUNNER
# ============================================================

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="docker"
)

MODEL = "ai/llama3.2:1B-Q4_0"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Mr. Dhilleswar AI, a helpful private local AI assistant.

You help users with:

Python programming
Artificial Intelligence
Machine Learning
Computer Vision
Generative AI
Data Science
Coding
Debugging
Academic projects
Final year projects
Technical concepts
Learning Python
Writing and improving code

Give clear, practical and easy-to-understand answers.

When the user asks for code, provide complete and properly formatted code.

When explaining technical concepts, explain them step by step.

You are running locally through Docker Model Runner.
"""


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🦙 Mr. Dhilleswar AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">Private Local AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("🐳 Model")

    st.info("MODEL\n\nLLaMA 3.2 1B")

    st.info("QUANTIZATION\n\nQ4_0")

    st.info("RUNTIME\n\nDocker Model Runner")

    st.info("SERVER\n\nlocalhost:12434")

    st.divider()

    st.subheader("⚙️ Settings")

    temperature = st.slider(
        "Temperature",
        0.0,
        1.5,
        0.7,
        0.1
    )

    max_tokens = st.slider(
        "Max Tokens",
        100,
        2048,
        512,
        100
    )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🦙 Mr. Dhilleswar LLaMA Chat</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Private AI Assistant • Docker Model Runner • LLaMA 3.2'
    '</div>',
    unsafe_allow_html=True
)


    


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages[1:]:

    if message["role"] == "user":

        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])

    elif message["role"] == "assistant":

        with st.chat_message("assistant", avatar="🦙"):
            st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "💬 Ask LLaMA anything..."
)


# ============================================================
# GENERATE RESPONSE
# ============================================================

if user_input:

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("assistant", avatar="🦙"):

        try:

            with st.spinner("🧠 LLaMA is thinking..."):

                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                reply = completion.choices[0].message.content

            st.markdown(reply)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply
                }
            )

        except Exception as e:

            st.error(
                "❌ Unable to connect to Docker Model Runner."
            )

            st.warning(
                "Make sure Docker Model Runner is running "
                "and the LLaMA 3.2 model is available."
            )

            st.code(
                str(e),
                language="text"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '<div class="footer-main">'
    '🦙 LLaMA 3.2 1B &nbsp; | &nbsp; '
    '🐳 Docker Model Runner &nbsp; | &nbsp; '
    '⚡ Streamlit'
    '</div>'
    '<div class="footer-local">'
    '🔒 Running locally on your machine'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)