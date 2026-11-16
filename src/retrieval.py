"""Validation helpers for embeddings and vector-search results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any


class EmbeddingValidationError(ValueError):
    """Raised when an embedding cannot be safely stored or compared."""


def _as_flat_list(embedding: Any) -> list[Any]:
    """Convert common tensor/array outputs into one flat Python list."""

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().tolist()
    elif hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise EmbeddingValidationError("Embedding must be a one-dimensional sequence")

    values = list(embedding)
    if values and isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        raise EmbeddingValidationError("A single embedding was expected, not a batch")
    return values


def validate_embedding(
    embedding: Any,
    *,
    expected_dims: int | None = None,
    expected_model: str | None = None,
    actual_model: str | None = None,
    require_normalized: bool = True,
    tolerance: float = 1e-3,
) -> list[float]:
    """Validate and return an embedding as a flat list of floats.

    Checks cover shape, numeric/finiteness safety, dimensions, non-zero norm,
    optional unit normalization, and model identity.
    """

    values = _as_flat_list(embedding)
    if not values:
        raise EmbeddingValidationError("Embedding cannot be empty")
    if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
        raise EmbeddingValidationError("Embedding values must be numeric")

    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingValidationError("Embedding contains NaN or infinite values")

    if expected_dims is not None and len(vector) != expected_dims:
        raise EmbeddingValidationError(
            f"Embedding dimension mismatch: expected {expected_dims}, got {len(vector)}"
        )
    if expected_model is not None and actual_model is not None and expected_model != actual_model:
        raise EmbeddingValidationError(
            f"Embedding model mismatch: expected '{expected_model}', got '{actual_model}'"
        )

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise EmbeddingValidationError("Embedding has zero norm")
    if require_normalized and not math.isclose(norm, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise EmbeddingValidationError(
            f"Embedding is not normalized: L2 norm is {norm:.6f}, expected 1.0"
        )

    return vector


def validate_embedding_document(
    document: dict[str, Any],
    *,
    expected_model: str | None = None,
    expected_dims: int | None = None,
    require_normalized: bool = True,
) -> list[float]:
    """Validate an embedding stored in a MongoDB document."""

    if "embedding" not in document:
        raise EmbeddingValidationError("Document has no embedding field")

    stored_model = document.get("model", document.get("embedding_model"))
    stored_dims = document.get("dims", document.get("embedding_dims"))
    effective_dims = expected_dims if expected_dims is not None else stored_dims

    return validate_embedding(
        document["embedding"],
        expected_dims=effective_dims,
        expected_model=expected_model,
        actual_model=stored_model,
        require_normalized=require_normalized,
    )


def validate_embedding_pair(
    query_embedding: Any,
    document_embedding: Any,
    *,
    expected_dims: int | None = None,
    require_normalized: bool = True,
) -> tuple[list[float], list[float]]:
    """Validate query/document vectors before calculating similarity."""

    query = validate_embedding(
        query_embedding,
        expected_dims=expected_dims,
        require_normalized=require_normalized,
    )
    document = validate_embedding(
        document_embedding,
        expected_dims=expected_dims,
        require_normalized=require_normalized,
    )
    if len(query) != len(document):
        raise EmbeddingValidationError(
            f"Cannot compare embeddings with dimensions {len(query)} and {len(document)}"
        )
    return query, document


def cosine_similarity(
    left: Any,
    right: Any,
    *,
    expected_dims: int | None = None,
) -> float:
    """Calculate cosine similarity after validating both vectors."""

    left_vector, right_vector = validate_embedding_pair(
        left,
        right,
        expected_dims=expected_dims,
        require_normalized=False,
    )
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    return sum(a * b for a, b in zip(left_vector, right_vector)) / (left_norm * right_norm)


def validate_retrieval_results(results: list[dict[str, Any]], expected_k: int | None = None) -> None:
    """Check the basic integrity of results returned to the RAG prompt."""

    if expected_k is not None and len(results) != expected_k:
        raise EmbeddingValidationError(
            f"Expected {expected_k} retrieval results, got {len(results)}"
        )

    scores: list[float] = []
    chunk_ids: set[Any] = set()
    for result in results:
        if "chunk_id" not in result or "text" not in result or "score" not in result:
            raise EmbeddingValidationError(
                "Every retrieval result must contain chunk_id, text, and score"
            )
        if not isinstance(result["text"], str) or not result["text"].strip():
            raise EmbeddingValidationError("Retrieval result text cannot be empty")
        score = result["score"]
        if not isinstance(score, Real) or not math.isfinite(float(score)):
            raise EmbeddingValidationError("Retrieval score must be finite and numeric")
        if result["chunk_id"] in chunk_ids:
            raise EmbeddingValidationError("Retrieval results contain duplicate chunk ids")
        chunk_ids.add(result["chunk_id"])
        scores.append(float(score))

    if scores != sorted(scores, reverse=True):
        raise EmbeddingValidationError("Retrieval results are not sorted by descending score")


class Retriever:
    """Embed user queries and delegate exact/approximate search to a vector store."""

    def __init__(
        self,
        vector_store: Any,
        embedding_model: Any,
        model_name: str,
        vector_index: str = "vector_index",
    ):
        """Configure retrieval with an embedding model and vector index."""

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.vector_index = vector_index
        modern_method = getattr(embedding_model, "get_embedding_dimension", None)
        legacy_method = getattr(embedding_model, "get_sentence_embedding_dimension", None)
        self.embedding_dims = (
            modern_method() if modern_method is not None else legacy_method()
        )

    def embed_query(self, query: str) -> list[float]:
        """Encode and validate a user query using the configured embedding model."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        return validate_embedding(
            embedding,
            expected_dims=self.embedding_dims,
            expected_model=self.model_name,
            actual_model=self.model_name,
            require_normalized=True,
        )

    def exact_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Run exact k-nearest-neighbor search over the stored vectors."""

        return self.vector_store.exact_search(
            self.embed_query(query),
            k=k,
            expected_dims=self.embedding_dims,
            expected_model=self.model_name,
        )

    def vector_search(self, query: str, k: int = 5, **filters: Any) -> list[dict[str, Any]]:
        """Run MongoDB Atlas approximate vector search for a user query."""

        return self.vector_store.vector_search(
            self.embed_query(query),
            k=k,
            index_name=self.vector_index,
            **filters,
        )
