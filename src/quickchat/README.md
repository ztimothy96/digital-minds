# quickchat

Exploratory-prompting UI: pick an open-weights HF model, set a system prompt,
have a multi-turn conversation, and log the transcript. Inference runs via
the HuggingFace Inference API, so there's no deploy step and no local GPU
needed — this app is for fast poking-around only. The probing/steering
experimental harness is a separate, Modal-backed pipeline (needed there for
activation access), not this one.

## Setup

From the repo root (installs this and other subprojects together via the
uv workspace):

```bash
uv sync
```

Or standalone, from this directory:

```bash
pip install -e .
```

Set an HF token (needed for gated models and to raise rate limits on the
free tier) either as an environment variable or by pasting it into the
sidebar at runtime:

```bash
export HF_TOKEN=hf_...
```

## Run

From this directory:

```bash
streamlit run quickchat/app.py
```

Or from the repo root, via the uv workspace:

```bash
uv run streamlit run src/quickchat/quickchat/app.py
```

Transcripts are saved as JSON to `logs/` (gitignored), one file per
conversation, overwritten after each turn.

## Notes

- HF's Inference API routes each model through third-party "Inference
  Providers"; a model only works if at least one provider serving it is
  enabled on your account (huggingface.co/settings/inference-providers).
  Some curated models are only served by a single provider
  (`featherless-ai`), which isn't enabled by default — the app flags these
  in the sidebar. If you hit `"not supported by any provider you have
  enabled"`, either enable that provider or pick a model with broader
  provider support (e.g. Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct,
  gemma-3-4b-it).
- Model availability changes over time — if a curated model 404s or errors,
  check its `inferenceProviderMapping` at
  `huggingface.co/api/models/<id>?expand=inferenceProviderMapping`, or try
  another model.
- Gated models (Llama, Gemma) need a token that has accepted the model's
  license on huggingface.co.
