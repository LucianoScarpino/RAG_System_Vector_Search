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
        recreate: bool = False,
    ) -> str:
        """Create, reuse, or explicitly recreate a reproducible vector index."""

        expected_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": embeddings_path,
                    "numDimensions": embedding_dims,
                    "similarity": "cosine",
                },
                *({"type": "filter", "path": path} for path in filter_paths),
            ]
        }
        existing_indexes = list(self.collection.list_search_indexes(index_name))
        existing = next(
            (index for index in existing_indexes if index.get("name") == index_name),
            None,
        )

        if existing is not None and self._index_matches(existing, expected_definition):
            self._wait_for_index(index_name, timeout_seconds)
            return "reused"

        if existing is not None and not recreate:
            raise RuntimeError(
                f"Vector index '{index_name}' exists but is incompatible with the "
                "current embedding configuration. Re-run with the explicit "
                "recreate-index option to replace it."
            )

        if existing is not None:
            self.collection.drop_search_index(index_name)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if not list(self.collection.list_search_indexes(index_name)):
                    break
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"Existing vector index '{index_name}' was not removed in time"
                )

        model = SearchIndexModel(
            definition=expected_definition,
            name=index_name,
            type="vectorSearch",
        )
        self.collection.create_search_index(model=model)
        self._wait_for_index(index_name, timeout_seconds)
        return "recreated" if existing is not None else "created"

    @staticmethod
    def _index_matches(index: dict, expected_definition: dict) -> bool:
        """Compare the relevant vector-index definition fields."""

        actual_definition = index.get("latestDefinition") or index.get("definition")
        if not actual_definition:
            return False

        def normalize(fields: list[dict]) -> list[tuple]:
            """Normalize index fields so ordering does not affect comparison."""

            return sorted(
                (
                    field.get("type"),
                    field.get("path"),
                    field.get("numDimensions"),
                    field.get("similarity"),
                )
                for field in fields
            )

        return normalize(actual_definition.get("fields", [])) == normalize(
            expected_definition.get("fields", [])
        )

    def _wait_for_index(self, index_name: str, timeout_seconds: int) -> None:
        """Wait until an Atlas vector index reports that it is queryable."""

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            indexes = list(self.collection.list_search_indexes(index_name))
            if any(index.get("queryable") for index in indexes):
                return
            time.sleep(5)
        raise TimeoutError(f"Vector index '{index_name}' did not become queryable in time")

    def wait_for_vector_results(
        self,
        query_embedding: list[float],
        k: int = 1,
        timeout_seconds: int = 120,
        poll_seconds: int = 5,
        **search_kwargs: Any,
    ) -> list[dict]:
        """Wait for Atlas to return results after newly embedded documents are written."""

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            results = self.vector_search(query_embedding, k=k, **search_kwargs)
            if results:
                return results
            time.sleep(poll_seconds)
        raise TimeoutError(
            "The vector index is queryable but has not returned results for the "
            "newly indexed documents within the timeout."
        )

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
