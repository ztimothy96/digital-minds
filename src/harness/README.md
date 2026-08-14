# harness

Probing/steering pipeline: build a contrastive (positive/negative) persona
dataset for a trait, train a [repeng](https://github.com/vgel/repeng)
steering vector on Modal, then compare baseline vs. steered generations and
probe arbitrary text against the trained direction. Runs on Modal — unlike
`quickchat`'s HF-Inference-API path, repeng needs the actual model weights
loaded with hooks on specific layers, which the Inference API can't give us.

## How it works

1. **Dataset** (`harness/traits.py`): a shared instruction template is
   filled with a positive vs. negative persona word (e.g. "honest" vs.
   "deceptive"), followed by a shared suffix, so each pair differs only in
   the persona framing — the same recipe repeng's own notebooks use. A
   small bank of topic-diverse suffixes is truncated at several lengths to
   multiply into a larger dataset.
2. **Training** (`harness/modal_app.py`): `ControlVector.train()` runs a
   forward pass over every pair, takes the last-token hidden state at each
   middle-to-late layer (25%–75% depth — repeng's own examples steer in
   roughly this band), and fits a 1D PCA direction per layer.
3. **Steering**: `ControlModel` adds `coeff * direction` to the residual
   stream at those layers during generation.
4. **Probing**: projecting a new text's activations onto the trained
   direction gives a scalar "how much does this text express the trait"
   score — the same underlying vector serves both jobs.

## Setup

From the repo root (installs this and other subprojects together via the
uv workspace):

```bash
uv sync
modal setup   # one-time Modal account auth, if you haven't already
```

Gated models (Llama, Gemma) additionally need an HF token with the license
accepted on huggingface.co, stored as a Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

## Run

```bash
modal deploy harness/modal_app.py   # deploy training/steering backend (once, and after edits)
streamlit run harness/app.py
```

If you created the `huggingface-secret` above, deploy with it enabled:

```bash
HARNESS_HF_SECRET=1 modal deploy harness/modal_app.py
```

Transcripts (prompt + baseline + steered response per turn) are saved as
JSON to `logs/` (gitignored), one file per session, overwritten after each
turn.

Trained vectors only live in the Streamlit session until you click "Save
vector to disk" in the first tab, which writes them to `vectors/`
(gitignored) as a metadata JSON + pickled `ControlVector` bytes. Reload a
saved vector from the same tab instead of retraining.

## Notes

- The chat/compare tab is single-turn by design: each prompt is generated
  independently against baseline and steered, rather than threading a
  multi-turn conversation, so the two variants never diverge in confusing
  ways. Use `quickchat` for multi-turn exploration.
- Trained direction vectors are unit-norm (PCA components), while residual-
  stream activation norms vary by model and layer (the compare tab shows
  the trained vector's typical activation norm for reference). If a trait
  doesn't seem to be having an effect, try some different coefficient
  values before concluding it "doesn't work."
- The default trait bank (honesty, contentment, corrigibility, agency) is
  chosen for this project's personality/welfare focus; add more in
  `harness/traits.py`, or use the "custom" option in the UI.
