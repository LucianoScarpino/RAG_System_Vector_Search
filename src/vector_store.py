"""MongoDB Atlas storage and vector-search operations."""

from __future__ import annotations

import time
from typing import Any

from pymongo.operations import SearchIndexModel

from src.retrieval import (
    cosine_similarity,
    validate_embedding_document,
    validate_retrieval_results,
)


class VectorStore:
    """Thin repository layer around a MongoDB collection of chunks."""

    def __init__(self, collection):
        """Wrap a MongoDB collection containing chunk documents."""

        self.collection = collection

    @classmethod
    def from_client(cls, client, database: str, collection: str) -> "VectorStore":
        """Create a store from a MongoDB client and collection names."""

        return cls(client[database][collection])

    def insert_chunks(self, chunks: list[dict], replace_document_id: str | None = None) -> int:
        """Insert chunks, optionally replacing chunks for one document id."""

        if replace_document_id is not None:
            self.collection.delete_many({"document_id": replace_document_id})
        if not chunks:
            return 0
        result = self.collection.insert_many(chunks)
        return len(result.inserted_ids)

    def get_chunk(self, chunk_id: int) -> dict | None:
        """Return one chunk by id without exposing MongoDB's internal id."""

        return self.collection.find_one({"chunk_id": chunk_id}, {"_id": 0})

    def keyword_search(self, keyword: str, limit: int = 10) -> list[dict]:
        """Find chunks containing a case-insensitive keyword or phrase."""

        if not keyword or limit <= 0:
            return []
        return list(
            self.collection.find(
                {"text": {"$regex": keyword, "$options": "i"}},
                {"_id": 0},
            ).limit(limit)
        )

    def statistics(self) -> list[dict]:
        """Return chunk-size fields for diagnostics and visualization."""

        return list(
            self.collection.find(
                {},
                {
                    "_id": 0,
                    "chunk_id": 1,
                    "chunk_token_count": 1,
                    "chunk_sentence_count": 1,
                    "chunk_char_count": 1,
                    "chunk_word_count": 1,
                },
            )
        )

    def exact_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        *,
        expected_dims: int | None = None,
        expected_model: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Run validated exact cosine search over all matching documents."""

        if k <= 0:
            raise ValueError("k must be positive")

        cursor = self.collection.find(
            filters or {},
            {"_id": 0, "embedding": 1, "chunk_id": 1, "text": 1, "chapter": 1, "model": 1, "dims": 1},
        )
        results = []
        for document in cursor:
            vector = validate_embedding_document(
                document,
                expected_model=expected_model,
                expected_dims=expected_dims,
            )
            score = cosine_similarity(query_embedding, vector, expected_dims=expected_dims)
            document.pop("embedding", None)
            document["score"] = score
            results.append(document)

        results.sort(key=lambda item: item["score"], reverse=True)
        results = results[:k]
        validate_retrieval_results(results)
        return results

    def create_vector_index(
        self,
        index_name: str,
        embedding_dims: int,
        embeddings_path: str = "embedding",
        filter_paths: tuple[str, ...] = ("chapter", "document_id"),
        timeout_seconds: int = 300,
    ) -> None:
        """Create and wait for a MongoDB Atlas vector-search index."""

        fields = [
            {
                "type": "vector",
                "path": embeddings_path,
                "numDimensions": embedding_dims,
                "similarity": "cosine",
            }
        ]
        fields.extend({"type": "filter", "path": path} for path in filter_paths)
        model = SearchIndexModel(
            definition={"fields": fields},
            name=index_name,
            type="vectorSearch",
        )

        if not list(self.collection.list_search_indexes(index_name)):
            self.collection.create_search_index(model=model)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if any(index.get("queryable") for index in self.collection.list_search_indexes(index_name)):
                return
            time.sleep(5)
        raise TimeoutError(f"Vector index '{index_name}' did not become queryable in time")

    def vector_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        *,
        index_name: str = "vector_index",
        embeddings_path: str = "embedding",
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Run validated approximate nearest-neighbor search in Atlas."""

        if k <= 0:
            raise ValueError("k must be positive")

        vector_stage: dict[str, Any] = {
            "index": index_name,
            "path": embeddings_path,
            "queryVector": query_embedding,
            "numCandidates": max(20 * k, 100),
            "limit": k,
        }
        if filters:
            vector_stage["filter"] = filters

        results = list(
            self.collection.aggregate(
                [
                    {"$vectorSearch": vector_stage},
                    {
                        "$project": {
                            "_id": 0,
                            "chunk_id": 1,
                            "text": 1,
                            "chapter": 1,
                            "document_id": 1,
                            "score": {"$meta": "vectorSearchScore"},
                        }
                    },
                ]
            )
        )
        validate_retrieval_results(results)
        return results
