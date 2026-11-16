import pytest

from src.vector_store import VectorStore


class FakeCollection:
    """Small in-memory collection double for vector-index lifecycle tests."""

    def __init__(self, indexes: list[dict]):
        """Initialize the fake collection with existing Atlas index metadata."""

        self.indexes = indexes
        self.created_models = []
        self.dropped_names = []

    def list_search_indexes(self, name: str | None = None) -> list[dict]:
        """Return the configured index metadata."""

        if name is None:
            return list(self.indexes)
        return [index for index in self.indexes if index.get("name") == name]

    def drop_search_index(self, name: str) -> None:
        """Remove one index from the fake Atlas collection."""

        self.dropped_names.append(name)
        self.indexes = [index for index in self.indexes if index.get("name") != name]

    def create_search_index(self, model) -> None:
        """Create a queryable fake index from a SearchIndexModel."""

        self.created_models.append(model)
        self.indexes.append(
            {
                "name": model.document["name"],
                "latestDefinition": model.document["definition"],
                "queryable": True,
            }
        )


def index_definition(dimension: int = 3, include_document_filter: bool = True) -> dict:
    """Return a representative Atlas vector-index definition."""

    fields = [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": dimension,
            "similarity": "cosine",
        },
        {"type": "filter", "path": "chapter"},
    ]
    if include_document_filter:
        fields.append({"type": "filter", "path": "document_id"})
    return {"fields": fields}


def test_compatible_vector_index_is_reused(monkeypatch):
    """Verify that a compatible existing index is reused without replacement."""

    collection = FakeCollection(
        [
            {
                "name": "vector_index",
                "latestDefinition": index_definition(),
                "queryable": True,
            }
        ]
    )
    store = VectorStore(collection)
    monkeypatch.setattr(store, "_wait_for_index", lambda *_args: None)

    action = store.create_vector_index("vector_index", embedding_dims=3)

    assert action == "reused"
    assert collection.created_models == []
    assert collection.dropped_names == []


def test_incompatible_vector_index_requires_explicit_recreation(monkeypatch):
    """Verify that an incompatible index fails without an explicit override."""

    collection = FakeCollection(
        [
            {
                "name": "vector_index",
                "latestDefinition": index_definition(include_document_filter=False),
                "queryable": True,
            }
        ]
    )
    store = VectorStore(collection)

    with pytest.raises(RuntimeError, match="incompatible"):
        store.create_vector_index("vector_index", embedding_dims=3)


def test_incompatible_vector_index_is_recreated_when_requested(monkeypatch):
    """Verify that explicit recreation replaces an incompatible index."""

    collection = FakeCollection(
        [
            {
                "name": "vector_index",
                "latestDefinition": index_definition(dimension=3),
                "queryable": True,
            }
        ]
    )
    store = VectorStore(collection)
    monkeypatch.setattr(store, "_wait_for_index", lambda *_args: None)

    action = store.create_vector_index(
        "vector_index",
        embedding_dims=4,
        recreate=True,
    )

    assert action == "recreated"
    assert collection.dropped_names == ["vector_index"]
    assert len(collection.created_models) == 1
    assert collection.indexes[0]["latestDefinition"] == index_definition(dimension=4)
