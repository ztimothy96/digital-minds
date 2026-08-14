"""Curated open-weight instruct models for the probing/steering harness.

Unlike quickchat's list (which is constrained by HF Inference Providers
availability), this harness loads full weights directly via `transformers`
on Modal, so the only constraint is gated-repo license acceptance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    params: str
    gated: bool = False


CURATED_MODELS: list[ModelOption] = [
    ModelOption("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"),
    ModelOption("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ModelOption("Qwen/Qwen2.5-3B-Instruct", "3B"),
    ModelOption("Qwen/Qwen2.5-7B-Instruct", "7B"),
    ModelOption("meta-llama/Llama-3.2-3B-Instruct", "3B", gated=True),
    ModelOption("meta-llama/Llama-3.1-8B-Instruct", "8B", gated=True),
    ModelOption("google/gemma-3-1b-it", "1B", gated=True),
    ModelOption("google/gemma-3-4b-it", "4B", gated=True),
]


def label(option: ModelOption) -> str:
    tag = " (gated)" if option.gated else ""
    return f"{option.model_id} — {option.params}{tag}"
