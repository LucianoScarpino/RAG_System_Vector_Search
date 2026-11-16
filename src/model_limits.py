"""Resolve the token budget of a Hugging Face embedding model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ChunkSizeMismatchError(ValueError):
    """Raised when the user rejects a manual/model chunk-size mismatch."""


@dataclass(frozen=True)
class ModelChunkingConfig:
    """Model and token budget needed by the token-aware chunker."""

    model_name: str
    model: Any
    tokenizer: Any
    model_max_tokens: int
    chunk_size: int


def _load_default_model(model_name: str, **model_kwargs: Any) -> Any:
    """Load a Sentence Transformer lazily, keeping imports optional for tests."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers to resolve Hugging Face model limits."
        ) from exc

    return SentenceTransformer(model_name, **model_kwargs)


def _read_max_tokens(model: Any) -> int:
    """Read the effective input limit used by Sentence Transformers."""

    max_tokens = getattr(model, "max_seq_length", None)
    tokenizer = getattr(model, "tokenizer", None)

    if max_tokens is None and tokenizer is not None:
        max_tokens = getattr(tokenizer, "model_max_length", None)

    # Hugging Face uses a very large sentinel when no concrete tokenizer limit
    # is defined.  Fall back to the model configuration in that case.
    if max_tokens is not None and max_tokens >= 10**9:
        config = getattr(model, "config", None)
        max_tokens = getattr(config, "max_position_embeddings", None)

    if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens >= 10**9:
        raise ValueError(
            "Unable to determine a finite maximum token length for the model."
        )

    return max_tokens


def get_model_chunking_config(
    model_name: str,
    manual_chunk_size: int | None = None,
    confirm_fn: Callable[[str], str] = input,
    model_loader: Callable[..., Any] | None = None,
    **model_kwargs: Any,
) -> ModelChunkingConfig:
    """Load a model and resolve the token budget used by chunking.

    If ``manual_chunk_size`` is omitted, the model's effective maximum is used.
    If a manual value differs, the user is asked whether to continue. Rejecting
    the prompt raises :class:`ChunkSizeMismatchError` and stops the pipeline.
    """

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty Hugging Face model name")
    if manual_chunk_size is not None and (
        not isinstance(manual_chunk_size, int)
        or isinstance(manual_chunk_size, bool)
        or manual_chunk_size <= 0
    ):
        raise ValueError("manual_chunk_size must be a positive integer")

    loader = model_loader or _load_default_model
    model = loader(model_name, **model_kwargs)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("The selected model does not expose a tokenizer")

    model_max_tokens = _read_max_tokens(model)
    chunk_size = model_max_tokens

    if manual_chunk_size is not None and manual_chunk_size != model_max_tokens:
        answer = confirm_fn(
            f"The model '{model_name}' supports at most {model_max_tokens} tokens, "
            f"but the requested chunk size is {manual_chunk_size}. Continue? [y/N] "
        )
        if str(answer).strip().lower() not in {"y", "yes"}:
            raise ChunkSizeMismatchError(
                "Chunking stopped: the manual chunk size does not match "
                "the model maximum token length."
            )
        chunk_size = manual_chunk_size
    elif manual_chunk_size is not None:
        chunk_size = manual_chunk_size

    return ModelChunkingConfig(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        model_max_tokens=model_max_tokens,
        chunk_size=chunk_size,
    )


def get_model_max_tokens(
    model_name: str,
    model_loader: Callable[..., Any] | None = None,
    **model_kwargs: Any,
) -> int:
    """Return the effective maximum number of input tokens for an embedding model."""

    return get_model_chunking_config(
        model_name=model_name,
        model_loader=model_loader,
        **model_kwargs,
    ).model_max_tokens
