"""Prompt construction and Hugging Face text generation."""

from __future__ import annotations

from typing import Any


ABSTENTION = "I don't have enough information."
RULES = (
    "Answer the question using ONLY the context below. "
    f"If the context does not contain the answer, say exactly: \"{ABSTENTION}\" "
    "Cite the chunk ids you used. Do not invent facts."
)


def format_prompt(query: str, results: list[dict[str, Any]]) -> str:
    """Build the grounded prompt used by the generator."""

    context = "\n".join(
        f"[chunk {result['chunk_id']}] {result['text']}" for result in results
    )
    return f"{RULES}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"


class TextGenerator:
    """Lazy wrapper around a Hugging Face text-generation pipeline."""

    def __init__(
        self,
        model_id: str,
        device: int | str | None = None,
        max_new_tokens: int = 300,
        **pipeline_kwargs: Any,
    ):
        """Configure a lazy Hugging Face text-generation pipeline."""

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.pipeline_kwargs = pipeline_kwargs
        self.device = device
        self._generator = None

    @property
    def generator(self):
        """Load and cache the Hugging Face pipeline on first use."""

        if self._generator is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "Install transformers and torch to generate answers."
                ) from exc

            if self.device is None:
                self.device = 0 if torch.cuda.is_available() else -1
            self._generator = pipeline(
                "text-generation",
                model=self.model_id,
                device=self.device,
                **self.pipeline_kwargs,
            )
        return self._generator

    @staticmethod
    def _extract_text(output: Any) -> str:
        """Extract assistant text from common text-generation output formats."""

        generated = output[0]["generated_text"] if isinstance(output, list) else output
        if isinstance(generated, list):
            last = generated[-1]
            return last.get("content", str(last)) if isinstance(last, dict) else str(last)
        return str(generated)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate one response for a non-empty prompt."""

        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        output = self.generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        return self._extract_text(output)
