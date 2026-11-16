"""Embedding model loading and batch generation."""

from __future__ import annotations

from typing import Any

import torch
from pymongo import UpdateOne
from tqdm.auto import tqdm

from src.retrieval import validate_embedding


class Embedding:
    """Create and persist normalized sentence embeddings."""

    @staticmethod
    def set_embedding_model(model_name: str, device: str | None = None, **kwargs: Any):
        """Load a Sentence Transformer on the requested or best available device."""

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to generate embeddings."
            ) from exc

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return SentenceTransformer(model_name, device=device, **kwargs)

    @staticmethod
    def embedding_dimension(embedding_model: Any) -> int:
        """Return the output vector dimension exposed by an embedding model."""

        modern_method = getattr(embedding_model, "get_embedding_dimension", None)
        legacy_method = getattr(embedding_model, "get_sentence_embedding_dimension", None)
        dimension = modern_method() if modern_method is not None else legacy_method()
        if dimension is None:
            raise ValueError("The embedding model does not expose its output dimension")
        return int(dimension)

    @staticmethod
    def encode_query(embedding_model: Any, text: str) -> list[float]:
        """Encode and validate one normalized query embedding."""

        embedding = embedding_model.encode(text, normalize_embeddings=True)
        dimension = Embedding.embedding_dimension(embedding_model)
        return validate_embedding(
            embedding,
            expected_dims=dimension,
            require_normalized=True,
        )

    def perform_embedding(
        self,
        embedding_model: Any,
        model_name: str,
        text_batch: Any,
        collection: Any,
        bulk_size: int = 100,
    ) -> int:
        """Embed documents missing vectors and persist validated vectors in bulk."""

        """Embed all documents missing an embedding and update them in bulk."""

        if bulk_size <= 0:
            raise ValueError("bulk_size must be positive")

        documents_cursor = collection.find(
            {"embedding": {"$exists": False}},
            {"_id": 1, "text": 1},
        )
        dimension = self.embedding_dimension(embedding_model)
        updated = 0

        def embed_and_update(batch: list[dict]) -> None:
            """Encode one batch and write its validated vectors to MongoDB."""

            nonlocal updated
            texts = [document["text"] for document in batch]
            embeddings = embedding_model.encode(texts, normalize_embeddings=True)
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(embeddings)}"
                )

            updates = []
            for document, embedding in zip(batch, embeddings):
                vector = validate_embedding(
                    embedding,
                    expected_dims=dimension,
                    expected_model=model_name,
                    actual_model=model_name,
                    require_normalized=True,
                )
                updates.append(
                    UpdateOne(
                        {"_id": document["_id"]},
                        {
                            "$set": {
                                "embedding": vector,
                                "model": model_name,
                                "dims": dimension,
                                "embedding_normalized": True,
                            }
                        },
                    )
                )
            if updates:
                collection.bulk_write(updates)
                updated += len(updates)

        batch: list[dict] = []
        for document in tqdm(
            documents_cursor,
            total=collection.count_documents({"embedding": {"$exists": False}}),
            desc="Generating embeddings",
        ):
            batch.append(document)
            if len(batch) >= bulk_size:
                embed_and_update(batch)
                batch = []
        if batch:
            embed_and_update(batch)

        return updated
