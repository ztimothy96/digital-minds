"""Curated list of light/mid-weight, non-reasoning, decoder-only instruct
models. All are standard HF `transformers` architectures so activations stay
accessible for white-box work (probing, steering, TransformerLens) later.

Only models with at least one "live" entry in the HF Inference Providers
mapping are listed (checked against
https://huggingface.co/api/models/<id>?expand=inferenceProviderMapping).
Some models are only served by a single, non-default provider
(featherless-ai) that must be enabled manually at
https://huggingface.co/settings/inference-providers — `single_provider`
flags those so the UI can warn about it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    params: str
    gated: bool = False
    single_provider: str | None = None


CURATED_MODELS: list[ModelOption] = [
    ModelOption("Qwen/Qwen2.5-7B-Instruct", "7B"),
    ModelOption("meta-llama/Llama-3.1-8B-Instruct", "8B", gated=True),
    ModelOption("google/gemma-3-4b-it", "4B", gated=True),
]


def label(option: ModelOption) -> str:
    tag = " (gated)" if option.gated else ""
    return f"{option.model_id} — {option.params}{tag}"
