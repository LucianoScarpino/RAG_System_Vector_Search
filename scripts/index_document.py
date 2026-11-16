"""Index a PDF into MongoDB Atlas."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.chunking import Chunking
from src.config import HostConnection, Settings
from src.embeddings import Embedding
from src.ingestion import DataIngestion
from src.model_limits import get_model_chunking_config
from src.vector_store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for PDF indexing."""

    parser = argparse.ArgumentParser(description="Index a PDF for RAG retrieval")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the PDF")
    parser.add_argument("--document-id", help="Stable id; defaults to the PDF stem")
    parser.add_argument("--book-title", help="Display title; defaults to document id")
    parser.add_argument("--embedding-model", help="Hugging Face embedding model id")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Manual token budget; omitted means the model maximum",
    )
    parser.add_argument("--padding", type=int, default=1, help="Overlapping sentence units")
    parser.add_argument("--bulk-size", type=int, default=100)
    parser.add_argument("--database")
    parser.add_argument("--collection")
    parser.add_argument("--vector-index")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing chunks for this document id before indexing",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Recreate an existing vector index when its definition is incompatible",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept a manual/model chunk-size mismatch automatically",
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Merge command-line overrides into environment-based settings."""

    settings = Settings.from_env()
    changes = {}
    if args.embedding_model:
        changes["embedding_model"] = args.embedding_model
    if args.database:
        changes["database"] = args.database
    if args.collection:
        changes["collection"] = args.collection
    if args.vector_index:
        changes["vector_index"] = args.vector_index
    return replace(settings, **changes)


def index_document(args: argparse.Namespace) -> dict[str, int | str]:
    """Ingest, chunk, embed, store, and index one PDF document."""

    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    settings = _settings_from_args(args)
    document_id = args.document_id or args.pdf.stem
    book_title = args.book_title or document_id
    confirm_fn = (lambda _prompt: "y") if args.yes else input

    model_config = get_model_chunking_config(
        settings.embedding_model,
        manual_chunk_size=args.chunk_size,
        confirm_fn=confirm_fn,
    )

    ingestion = DataIngestion()
    chapters = ingestion.read_with_outline(args.pdf)
    nlp = ingestion.get_sentences()
    chunker = Chunking()

    chunks = []
    for chapter in chapters:
        chapter_chunks = chunker.chunk_text(
            chapter["text"],
            chunk_size=model_config.chunk_size,
            padding=args.padding,
            nlp=nlp,
            tokenizer=model_config.tokenizer,
        )
        for chunk in chapter_chunks:
            chunk.update(
                document_id=document_id,
                book=book_title,
                chapter=chapter["chapter"],
                chapter_title=chapter["chapter_title"],
            )
            chunks.append(chunk)

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    client = HostConnection.connect(settings.mongodb_uri)
    store = VectorStore.from_client(client, settings.database, settings.collection)
    dimensions = Embedding.embedding_dimension(model_config.model)
    index_action = store.create_vector_index(
        settings.vector_index,
        embedding_dims=dimensions,
        recreate=args.recreate_index,
    )
    inserted = store.insert_chunks(
        chunks,
        replace_document_id=document_id if args.replace else None,
    )

    embedding = Embedding()
    embedded = embedding.perform_embedding(
        model_config.model,
        settings.embedding_model,
        None,
        store.collection,
        bulk_size=args.bulk_size,
    )
    print("Waiting for Atlas to index the new vectors...", flush=True)
    store.wait_for_vector_results(
        Embedding.encode_query(model_config.model, "What is the document about?"),
        k=1,
        index_name=settings.vector_index,
    )
    result = {
        "document_id": document_id,
        "chapters": len(chapters),
        "chunks": inserted,
        "embedded": embedded,
        "max_model_tokens": model_config.model_max_tokens,
        "chunk_size": model_config.chunk_size,
        "vector_index_action": index_action,
    }
    print(result)
    return result


def main() -> None:
    """Parse command-line arguments and index the selected PDF."""

    index_document(build_parser().parse_args())


if __name__ == "__main__":
    main()
