# digital-minds
Apart Hackathon August 2026

Hackathon project exploring AI personality and welfare.

## Setup

This repo is a [uv](https://docs.astral.sh/uv/) workspace — one command
installs every subproject (`src/quickchat`, `src/jacobian-lens-main`) into a
shared environment at the repo root:

```bash
uv sync
uv run streamlit run src/quickchat/quickchat/app.py
```

Each subproject's own README has details on running it directly.

Starter code should include:
- Streamlit quick-chat interface where users can select open-source model of choice from HuggingFace, set system prompts (possibly empty), and have multi-turn conversations, and save the transcript to a log file.
- Experimental harness for probing and steering: given some dataset of posiive/negative examples, a pipeline collects activations, trains probes, and performs vector steering on 
- Interface for selecting a model and viewing J-lens top words, similar to what already exists in the Anthropic demo.