
import streamlit as st
import torch

from pathlib import Path

from generate import load_model
from assistant import ask, load_tokenizer


# =====================================
# 1. Page configuration
# =====================================

st.set_page_config(
    page_title="MiniGPT",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MiniGPT")

st.caption(
    "Your personal AI assistant built from scratch."
)


# =====================================
# 2. Load our latest model
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = Path(
    "checkpoints/minigpt_assistant_v2.pt"
)


@st.cache_resource(max_entries=1)
def get_model(path, modified_time):

    model = load_model(path, device)

    # Automatically load our custom V3 tokenizer
    tokenizer = load_tokenizer(model)

    return model, tokenizer


# Check that our trained model exists
if not checkpoint_path.exists():

    st.error(
        "MiniGPT V3 checkpoint not found. "
        "Run train_assistant_v3.py first."
    )

    st.stop()


model, tokenizer = get_model(
    str(checkpoint_path),
    checkpoint_path.stat().st_mtime
)


# =====================================
# 3. Sidebar
# =====================================

st.sidebar.title("MiniGPT")

st.sidebar.write("Model: Assistant V2")

parameter_count = sum(
    p.numel() for p in model.parameters()
)

st.sidebar.write(
    f"Parameters: {parameter_count:,}"
)

st.sidebar.write(
    "Device:", device.upper()
)

st.sidebar.write(
    "Vocabulary:",
    model.config["vocab_size"]
)

st.sidebar.write(
    "Transformer layers:",
    model.config["num_layers"]
)

st.sidebar.write(
    "Context length:",
    model.config["context_length"]
)

st.sidebar.success("Model loaded")

st.sidebar.divider()

max_tokens = st.sidebar.slider(
    "Maximum answer length",
    min_value=20,
    max_value=200,
    value=100,
    step=10
)

st.sidebar.caption(
    "An experimental AI assistant trained "
    "from scratch using modern English "
    "and instruction data."
)


# =====================================
# 4. Conversation history
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if st.sidebar.button(
    "New Chat",
    use_container_width=True
):

    st.session_state.messages = []

    st.rerun()


# Welcome message
if not st.session_state.messages:

    st.markdown("### How can I help you today?")

    st.caption(
        "Try asking about Python, AI, "
        "basic mathematics or general knowledge."
    )


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =====================================
# 5. Generate answers
# =====================================

if question := st.chat_input("Ask MiniGPT..."):

    # Display and save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = ask(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    max_new_tokens=max_tokens
                )

                if not answer.strip():

                    answer = (
                        "I couldn't generate an answer. "
                        "Try asking a different question."
                    )

            except Exception as error:

                answer = f"Generation error: {error}"

        st.markdown(answer)


    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
