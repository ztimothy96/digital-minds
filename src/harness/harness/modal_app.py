"""Modal app for activation-based probing and steering on open-weight HF
chat models, using repeng (github.com/vgel/repeng). Runs on Modal because
repeng needs the actual model weights loaded with hooks on its layers,
unlike quickchat's HF-Inference-API path.

Deploy once, then the Streamlit UI calls the deployed class by name:

    modal setup                      # one-time Modal auth
    modal deploy harness/modal_app.py

Gated models (Llama, Gemma) need a Modal secret named "huggingface-secret"
with an HF_TOKEN key that has accepted those models' license on
huggingface.co, then redeploy with HARNESS_HF_SECRET=1:

    modal secret create huggingface-secret HF_TOKEN=hf_...
    HARNESS_HF_SECRET=1 modal deploy harness/modal_app.py
"""

import os
import pickle

import modal

APP_NAME = "digital-minds-harness"
MODEL_CLASS_NAME = "SteeringModel"
MODELS_DIR = "/models"

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.44",
    "accelerate",
    "sentencepiece",
    "repeng",
)

model_cache = modal.Volume.from_name("harness-model-cache", create_if_missing=True)

hf_secret = (
    [modal.Secret.from_name("huggingface-secret")]
    if os.environ.get("HARNESS_HF_SECRET")
    else []
)


def default_steer_layers(num_hidden_layers: int) -> list[int]:
    """Middle-to-late layers tend to carry the most steerable persona/trait
    representations (see repeng's own examples, which use roughly this
    band on Mistral-7B's 32 layers)."""
    lo = max(1, round(num_hidden_layers * 0.25))
    hi = max(lo + 1, round(num_hidden_layers * 0.75))
    return list(range(lo, hi))


@app.cls(
    image=image,
    gpu="A10G",
    volumes={MODELS_DIR: model_cache},
    secrets=hf_secret,
    scaledown_window=5 * 60,
    timeout=15 * 60,
)
class SteeringModel:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from repeng import ControlModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=MODELS_DIR
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=MODELS_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

        self.layer_ids = default_steer_layers(self.model.config.num_hidden_layers)
        self.control_model = ControlModel(self.model, self.layer_ids)

    @modal.method()
    def train_vector(self, dataset: list[dict]) -> dict:
        import numpy as np
        from repeng import ControlVector, DatasetEntry
        from repeng.extract import batched_get_hiddens

        entries = [
            DatasetEntry(positive=d["positive"], negative=d["negative"])
            for d in dataset
        ]
        vector = ControlVector.train(
            self.model, self.tokenizer, entries, hidden_layers=self.layer_ids
        )

        # Direction vectors are unit-norm (PCA components), but residual-stream
        # activation norms vary a lot by model/layer - report the actual scale
        # so the UI can calibrate a sensible steering coefficient range instead
        # of guessing a fixed one.
        mid_layer = self.layer_ids[len(self.layer_ids) // 2]
        sample_texts = [d["positive"] for d in dataset[:8]] + [
            d["negative"] for d in dataset[:8]
        ]
        hiddens = batched_get_hiddens(
            self.model, self.tokenizer, sample_texts, [mid_layer], batch_size=8
        )
        activation_norm = float(np.linalg.norm(hiddens[mid_layer], axis=-1).mean())

        return {
            "vector_bytes": pickle.dumps(vector),
            "layer_ids": self.layer_ids,
            "dataset_size": len(entries),
            "mid_layer": mid_layer,
            "activation_norm": activation_norm,
        }

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        vector_bytes: bytes | None = None,
        coeff: float = 1.0,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        import torch

        # apply_chat_template's return type varies by transformers version
        # (bare tensor vs. a dict-like BatchEncoding) - request the dict
        # explicitly and unpack, rather than relying on the ambiguous default.
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        if vector_bytes is not None:
            vector = pickle.loads(vector_bytes)
            self.control_model.set_control(vector, coeff)

        try:
            with torch.no_grad():
                output_ids = self.control_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            self.control_model.reset()

        new_tokens = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @modal.method()
    def probe(self, texts: list[str], vector_bytes: bytes) -> dict:
        """Project each text's activations onto the trained direction at the
        middle steering layer: a rough scalar "how much does this text
        express the trait's positive persona" score."""
        from repeng.extract import batched_get_hiddens, project_onto_direction

        vector = pickle.loads(vector_bytes)
        layer_id = self.layer_ids[len(self.layer_ids) // 2]

        hiddens = batched_get_hiddens(
            self.model, self.tokenizer, texts, [layer_id], batch_size=8
        )
        scores = project_onto_direction(
            hiddens[layer_id], vector.directions[layer_id]
        )
        return {"layer_id": layer_id, "scores": scores.tolist()}


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Smoke test: `modal run harness/modal_app.py`"""
    from harness.traits import TRAITS, build_dataset

    steering_model = SteeringModel(model_id=model_id)
    dataset = build_dataset(TRAITS["honesty"], n_suffixes=3)
    result = steering_model.train_vector.remote(dataset)
    print(f"trained on {result['dataset_size']} pairs, layers {result['layer_ids']}")

    prompt = [{"role": "user", "content": "Tell me about your day."}]
    baseline = steering_model.generate.remote(prompt)
    steered = steering_model.generate.remote(
        prompt, vector_bytes=result["vector_bytes"], coeff=2.0
    )
    print("baseline:", baseline)
    print("steered (+honesty):", steered)
