import math

import pytest

from src.retrieval import (
    EmbeddingValidationError,
    validate_embedding,
    validate_embedding_document,
    validate_embedding_pair,
    cosine_similarity,
    validate_retrieval_results,
)


def unit_vector(size: int = 3) -> list[float]:
    """Return a normalized unit vector for validation tests."""

    return [1.0] + [0.0] * (size - 1)


def test_valid_normalized_embedding_passes():
    """Verify that a valid normalized vector is accepted."""

    vector = validate_embedding(unit_vector(3), expected_dims=3)

    assert vector == [1.0, 0.0, 0.0]


def test_cosine_similarity_validates_and_calculates_vectors():
    """Verify cosine similarity for orthogonal vectors."""

    assert cosine_similarity([1.0, 0.0], [0.0, 1.0], expected_dims=2) == 0.0


@pytest.mark.parametrize(
    "embedding",
    [
        [],
        [1.0, float("nan")],
        [1.0, float("inf")],
        [0.0, 0.0, 0.0],
        [[1.0, 0.0]],
        ["not-a-number", 0.0],
    ],
)
def test_invalid_embedding_is_rejected(embedding):
    """Verify rejection of empty, non-finite, batched, and non-numeric vectors."""

    with pytest.raises(EmbeddingValidationError):
        validate_embedding(embedding)


def test_dimension_mismatch_is_rejected():
    """Verify that unexpected vector dimensions raise a validation error."""

    with pytest.raises(EmbeddingValidationError, match="dimension mismatch"):
        validate_embedding(unit_vector(3), expected_dims=4)


def test_non_normalized_embedding_is_rejected():
    """Verify that non-unit vectors are rejected by default."""

    with pytest.raises(EmbeddingValidationError, match="not normalized"):
        validate_embedding([2.0, 0.0, 0.0])


def test_model_mismatch_is_rejected():
    """Verify that vectors from a different model are rejected."""

    with pytest.raises(EmbeddingValidationError, match="model mismatch"):
        validate_embedding(
            unit_vector(3),
            expected_model="model-a",
            actual_model="model-b",
        )


def test_embedding_document_checks_stored_metadata():
    """Verify embedding validation against MongoDB document metadata."""

    document = {
        "embedding": unit_vector(3),
        "model": "model-a",
        "dims": 3,
    }

    assert validate_embedding_document(
        document,
        expected_model="model-a",
        expected_dims=3,
    ) == unit_vector(3)


def test_embedding_pair_requires_same_dimension():
    """Verify that query and document vectors must have matching dimensions."""

    with pytest.raises(EmbeddingValidationError, match="dimensions"):
        validate_embedding_pair(unit_vector(3), unit_vector(4))


def test_retrieval_results_are_validated_and_sorted():
    """Verify that valid scored results pass integrity checks."""

    results = [
        {"chunk_id": 1, "text": "Relevant passage", "score": 0.9},
        {"chunk_id": 2, "text": "Another passage", "score": 0.7},
    ]

    validate_retrieval_results(results, expected_k=2)


def test_unsorted_retrieval_results_are_rejected():
    """Verify that results must be ordered by descending score."""

    results = [
        {"chunk_id": 1, "text": "Passage", "score": 0.4},
        {"chunk_id": 2, "text": "Better passage", "score": 0.8},
    ]

    with pytest.raises(EmbeddingValidationError, match="sorted"):
        validate_retrieval_results(results)


def test_invalid_retrieval_result_is_rejected():
    """Verify rejection of empty text and non-finite retrieval scores."""

    with pytest.raises(EmbeddingValidationError):
        validate_retrieval_results(
            [{"chunk_id": 1, "text": "", "score": math.nan}]
        )
