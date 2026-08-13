"""Thin wrapper around the HuggingFace Inference API for chat completion."""

import os

from huggingface_hub import InferenceClient


def get_client(token: str | None = None) -> InferenceClient:
    return InferenceClient(token=token or os.environ.get("HF_TOKEN"))


def generate(
    client: InferenceClient,
    model_id: str,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    completion = client.chat_completion(
        messages=messages,
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
    )
    return completion.choices[0].message.content
