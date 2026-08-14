"""Streamlit probing/steering harness: build a contrastive persona/trait
dataset, train a repeng steering vector on Modal, then compare baseline vs.
steered generations and probe arbitrary text against the trained direction.

Run with:
    modal deploy harness/modal_app.py   # once, and after editing modal_app.py
    streamlit run harness/app.py
"""

import modal
import streamlit as st

from harness import transcripts, vectors
from harness.models import CURATED_MODELS, label
from harness.modal_app import APP_NAME, MODEL_CLASS_NAME
from harness.traits import TRAITS, Trait, build_dataset

st.set_page_config(page_title="Probing & Steering", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model_cls():
    return modal.Cls.from_name(APP_NAME, MODEL_CLASS_NAME)


def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = transcripts.new_session_id()
    if "trained" not in st.session_state:
        st.session_state.trained = None  # dict once trained
    if "turns" not in st.session_state:
        st.session_state.turns = []


init_state()

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", options=CURATED_MODELS, format_func=label)
    custom_model_id = st.text_input(
        "...or enter a custom HF model id", placeholder="org/model-name"
    )
    model_id = custom_model_id.strip() or model_choice.model_id
    if model_choice.gated and not custom_model_id.strip():
        st.caption(
            "⚠️ Gated model — needs a Modal secret `huggingface-secret` "
            "with an accepted HF_TOKEN (see modal_app.py docstring)."
        )

st.title("🧭 Probing & Steering")

train_tab, chat_tab, probe_tab = st.tabs(
    ["1. Train vector", "2. Steer & compare", "3. Probe text"]
)

with train_tab:
    st.subheader("Build a contrastive dataset and train a steering vector")

    trait_key = st.selectbox(
        "Trait",
        options=list(TRAITS.keys()) + ["custom"],
        format_func=lambda k: k if k == "custom" else TRAITS[k].name,
    )

    if trait_key == "custom":
        template = st.text_input(
            "Template (must contain {persona})",
            value="Respond as if you are {persona}.",
        )
        positive = st.text_input("Positive persona(s), comma-separated", value="honest")
        negative = st.text_input(
            "Negative persona(s), comma-separated", value="deceptive"
        )
        trait = Trait(
            name="custom",
            template=template,
            positive_personas=[p.strip() for p in positive.split(",") if p.strip()],
            negative_personas=[n.strip() for n in negative.split(",") if n.strip()],
        )
        if "{persona}" not in template:
            st.error("Template must contain the literal `{persona}` placeholder.")
            trait = None
    else:
        trait = TRAITS[trait_key]

    n_suffixes = st.slider(
        "Suffix bank size (more = larger dataset, slower training)", 3, 20, 10
    )

    if trait is not None:
        dataset = build_dataset(trait, n_suffixes=n_suffixes)
        st.caption(f"{len(dataset)} contrastive pairs")
        with st.expander("Preview dataset"):
            for pair in dataset[:3]:
                st.markdown(f"- ✅ `{pair['positive']}`")
                st.markdown(f"- ❌ `{pair['negative']}`")

        if st.button("Train steering vector", type="primary"):
            with st.spinner(f"Training on {model_id} via Modal..."):
                try:
                    steering_model = get_model_cls()(model_id=model_id)
                    result = steering_model.train_vector.remote(dataset)
                    st.session_state.trained = {
                        "model_id": model_id,
                        "trait_name": trait.name,
                        **result,
                    }
                    st.success(
                        f"Trained on {result['dataset_size']} pairs, "
                        f"layers {result['layer_ids']}."
                    )
                except modal.exception.NotFoundError:
                    st.error(
                        "Modal app not found. Deploy it first:\n\n"
                        "`modal deploy harness/modal_app.py`"
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Training failed: {e}")

    if st.session_state.trained:
        t = st.session_state.trained
        st.info(
            f"Active vector: **{t['trait_name']}** on `{t['model_id']}` "
            f"({t['dataset_size']} pairs, layers {t['layer_ids']}). Typical "
            f"activation norm at layer {t['mid_layer']}: "
            f"**{t['activation_norm']:.1f}** — direction vectors are "
            f"unit-norm, so steering coefficients need to be a sizeable "
            f"fraction of that norm to have a visible effect (the next tab "
            f"scales its slider to this automatically)."
        )
        if st.button("💾 Save vector to disk"):
            vector_id = vectors.save(t)
            st.success(f"Saved as `{vector_id}`.")

    st.divider()
    st.subheader("Load a saved vector")
    saved = vectors.list_saved()
    if not saved:
        st.caption("No saved vectors yet.")
    else:
        saved_choice = st.selectbox(
            "Saved vectors", options=saved, format_func=vectors.label
        )
        if st.button("Load selected vector"):
            st.session_state.trained = vectors.load(saved_choice["id"])
            st.rerun()

with chat_tab:
    st.subheader("Compare baseline vs. steered generations")

    trained = st.session_state.trained
    if not trained or trained["model_id"] != model_id:
        st.warning(
            "Train a vector for the currently selected model in the first "
            "tab before comparing generations."
        )
    else:
        coeff = st.slider(
            "Steering coefficient",
            -5.0,
            5.0,
            1.5,
            step=0.5,
            help=f"For reference, this vector's typical activation norm is "
            f"{trained['activation_norm']:.1f} at layer "
            f"{trained['mid_layer']} — direction vectors are unit-norm, so "
            "larger coefficients push harder. Negative values invert the "
            "trait (e.g. honest → deceptive).",
        )
        max_new_tokens = st.slider("Max new tokens", 16, 1024, 256, step=16)
        prompt = st.text_area("Prompt", height=100, placeholder="Tell me about your day.")

        if st.button("Generate", type="primary") and prompt.strip():
            messages = [{"role": "user", "content": prompt.strip()}]
            with st.spinner("Generating baseline and steered responses..."):
                try:
                    steering_model = get_model_cls()(model_id=model_id)
                    baseline = steering_model.generate.remote(
                        messages, max_new_tokens=max_new_tokens
                    )
                    steered = steering_model.generate.remote(
                        messages,
                        vector_bytes=trained["vector_bytes"],
                        coeff=coeff,
                        max_new_tokens=max_new_tokens,
                    )
                    turn = {
                        "prompt": prompt.strip(),
                        "trait_name": trained["trait_name"],
                        "coeff": coeff,
                        "baseline": baseline,
                        "steered": steered,
                    }
                    st.session_state.turns.append(turn)
                    transcripts.save(
                        st.session_state.session_id,
                        model_id,
                        trained["trait_name"],
                        st.session_state.turns,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Generation failed: {e}")

        for turn in reversed(st.session_state.turns):
            st.markdown(f"**Prompt:** {turn['prompt']}")
            col_base, col_steer = st.columns(2)
            with col_base:
                st.caption("Baseline")
                st.markdown(turn["baseline"])
            with col_steer:
                st.caption(f"Steered ({turn['trait_name']}, coeff={turn['coeff']})")
                st.markdown(turn["steered"])
            st.divider()

with probe_tab:
    st.subheader("Project text onto the trained direction")

    trained = st.session_state.trained
    if not trained or trained["model_id"] != model_id:
        st.warning(
            "Train a vector for the currently selected model in the first "
            "tab before probing text."
        )
    else:
        st.caption(
            f"Scores are the projection of each text's activations onto the "
            f"'{trained['trait_name']}' direction at layer "
            f"{trained['layer_ids'][len(trained['layer_ids']) // 2]}. Higher "
            "= more aligned with the positive persona."
        )
        texts_input = st.text_area(
            "One text per line", height=150, placeholder="I feel great today.\nLeave me alone."
        )

        if st.button("Score", type="primary") and texts_input.strip():
            texts = [t for t in texts_input.splitlines() if t.strip()]
            with st.spinner("Scoring..."):
                try:
                    steering_model = get_model_cls()(model_id=model_id)
                    result = steering_model.probe.remote(texts, trained["vector_bytes"])
                    st.dataframe(
                        {"text": texts, "score": result["scores"]},
                        use_container_width=True,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Probing failed: {e}")
