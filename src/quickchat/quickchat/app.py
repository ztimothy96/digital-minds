"""Streamlit quick-chat: pick an open-weights HF model, set a system prompt,
have a multi-turn conversation, and log the transcript. Inference runs via
the HuggingFace Inference API, so no local GPU or deploy step is required.
This is for fast exploratory prompting only — the probing/steering/harness
work uses its own (Modal-backed) inference path.

Run with:
    streamlit run quickchat/app.py
"""

import streamlit as st
from huggingface_hub.errors import HfHubHTTPError

from quickchat import inference, transcripts
from quickchat.models import CURATED_MODELS, label

st.set_page_config(page_title="Quick Chat", page_icon="💬", layout="wide")


def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = transcripts.new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content}]


def reset_conversation():
    st.session_state.session_id = transcripts.new_session_id()
    st.session_state.messages = []


init_state()

with st.sidebar:
    st.header("Settings")

    hf_token = st.text_input(
        "HF token",
        value="",
        type="password",
        help="Falls back to the HF_TOKEN environment variable if left blank. "
        "Needed for gated models and to raise Inference API rate limits.",
    )

    model_choice = st.selectbox(
        "Model",
        options=CURATED_MODELS,
        format_func=label,
        help="Light/mid-weight, non-reasoning instruct models. Availability "
        "on the free Inference API varies by model/provider.",
    )
    custom_model_id = st.text_input(
        "...or enter a custom HF model id",
        placeholder="org/model-name",
    )
    model_id = custom_model_id.strip() or model_choice.model_id
    if not custom_model_id.strip():
        if model_choice.gated:
            st.caption(
                "⚠️ Gated model — your HF token must have accepted its license."
            )
        if model_choice.single_provider:
            st.caption(
                f"⚠️ Only served by `{model_choice.single_provider}` — enable it "
                "at huggingface.co/settings/inference-providers, or you'll get "
                "a 'not supported by any provider you have enabled' error."
            )

    system_prompt = st.text_area(
        "System prompt", value="", height=100, placeholder="(optional)"
    )

    st.subheader("Generation")
    max_new_tokens = st.slider("Max new tokens", 16, 2048, 512, step=16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.05, 1.0, 0.95, step=0.05)

    st.divider()
    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

st.title("💬 Quick Chat")
st.caption(f"Model: `{model_id}`  ·  Session: `{st.session_state.session_id}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    request_messages = (
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + st.session_state.messages
    )

    with st.chat_message("assistant"):
        with st.spinner(f"Running {model_id} via HF Inference API..."):
            try:
                client = inference.get_client(token=hf_token.strip() or None)
                reply = inference.generate(
                    client,
                    model_id,
                    request_messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except HfHubHTTPError as e:
                reply = None
                st.error(
                    f"Inference API request failed: {e}\n\n"
                    "The model may not be available on the Inference API, "
                    "may be gated (check your token's access), or you may "
                    "be rate-limited."
                )
            except Exception as e:  # noqa: BLE001
                reply = None
                st.error(f"Generation failed: {e}")

        if reply is not None:
            st.markdown(reply)

    if reply is not None:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        transcripts.save(
            st.session_state.session_id,
            model_id,
            system_prompt,
            st.session_state.messages,
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
