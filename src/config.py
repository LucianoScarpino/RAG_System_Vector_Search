"""Application configuration and MongoDB Atlas connection helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    mongodb_uri: str
    database: str = "rag_system"
    collection: str = "document_chunks"
    vector_index: str = "vector_index"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build application settings from environment variables."""

        uri = os.getenv("MONGODB_URI")
        if not uri:
            username = os.getenv("MONGODB_USERNAME")
            password = os.getenv("MONGODB_PASSWORD")
            host = os.getenv("MONGODB_CLUSTER_HOST")
            if username and password and host:
                uri = (
                    f"mongodb+srv://{quote_plus(username)}:{quote_plus(password)}"
                    f"@{host}/"
                )

        if not uri:
            raise RuntimeError(
                "Set MONGODB_URI or MONGODB_USERNAME, MONGODB_PASSWORD, "
                "and MONGODB_CLUSTER_HOST before connecting."
            )

        return cls(
            mongodb_uri=uri,
            database=os.getenv("MONGODB_DATABASE", "rag_system"),
            collection=os.getenv("MONGODB_COLLECTION", "document_chunks"),
            vector_index=os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
            ),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        )

    @classmethod
    def from_credentials_file(cls, path: str | Path) -> "Settings":
        """Load MongoDB credentials and optional database settings from JSON."""

        credentials_path = Path(path)
        try:
            payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Credentials JSON file not found: {credentials_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Credentials JSON file is not valid JSON: {credentials_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("Credentials JSON must contain one JSON object")

        uri = payload.get("mongodb_uri", payload.get("uri"))
        if uri is None:
            username = payload.get("username", payload.get("mongodb_username"))
            password = payload.get("password", payload.get("mongodb_password"))
            cluster_host = payload.get(
                "cluster_host",
                payload.get("mongodb_cluster_host"),
            )
            if not all(isinstance(value, str) and value for value in (username, password, cluster_host)):
                raise ValueError(
                    "Credentials JSON must contain mongodb_uri or username, "
                    "password, and cluster_host"
                )
            uri = (
                f"mongodb+srv://{quote_plus(username)}:{quote_plus(password)}"
                f"@{cluster_host}/"
            )

        if not isinstance(uri, str) or not uri.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(
                "The JSON MongoDB URI must start with 'mongodb://' or "
                "'mongodb+srv://'"
            )
        if "<" in uri or ">" in uri:
            raise ValueError("The JSON credentials still contain a placeholder value")

        return cls(
            mongodb_uri=uri,
            database=payload.get("database", "rag_system"),
            collection=payload.get("collection", "document_chunks"),
            vector_index=payload.get("vector_index", "vector_index"),
        )


class HostConnection:
    """Create an Atlas client and select the configured collection."""

    @staticmethod
    def connect(uri: str) -> MongoClient:
        """Connect to MongoDB Atlas and verify the connection with a ping."""

        client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())
        client.admin.command("ping")
        return client

    def connect_Atlas_client(
        self,
        db_username: str | None = None,
        db_password: str | None = None,
        cluster_host: str | None = None,
    ) -> MongoClient:
        """Connect using explicit credentials or the configured environment."""

        if db_username and db_password and cluster_host:
            uri = (
                f"mongodb+srv://{quote_plus(db_username)}:{quote_plus(db_password)}"
                f"@{cluster_host}/"
            )
        else:
            uri = Settings.from_env().mongodb_uri
        return self.connect(uri)

    @staticmethod
    def get_collection(client: MongoClient, settings: Settings):
        """Return the configured MongoDB collection from an open client."""

        return client[settings.database][settings.collection]

    def feed_collection(self, client: MongoClient, chunked_text: list[dict]):
        """Backward-compatible insertion helper used by the notebook workflow."""

        settings = Settings.from_env()
        collection = self.get_collection(client, settings)
        if chunked_text:
            collection.insert_many(chunked_text)
        return collection
