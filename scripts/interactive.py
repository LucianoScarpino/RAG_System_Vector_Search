"""Interactive setup, preflight checks, indexing, and query loop."""

from __future__ import annotations

import getpass
import os
from dataclasses import replace
from pathlib import Path

from src.chunking import Chunking
from src.config import HostConnection, Settings
from src.diagnostics import display_database_diagnostics
from src.embeddings import Embedding
from src.generation import TextGenerator
from src.ingestion import DataIngestion
from src.model_limits import get_model_chunking_config
from src.rag_pipeline import RAGPipeline
from src.retrieval import Retriever, validate_embedding
from src.vector_store import VectorStore


EXIT_COMMANDS = {"exit", "quit", "q", "esci", "fine"}


def _ask(prompt: str, default: str | None = None) -> str:
    """Read one value and explain explicitly how to accept its default."""

    if default is None:
        value = input(f"{prompt}: ").strip()
    else:
        value = input(f"{prompt} (press Enter to use '{default}'): ").strip()
    return value or (default or "")


def _ask_positive_int(prompt: str, default: int) -> int:
    """Read a positive integer from the terminal, retrying invalid input."""

    while True:
        raw = _ask(prompt, str(default))
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive integer.")


def _collect_settings() -> tuple[Settings, Path, int | None, int, bool]:
    """Collect connection, model, document, and chunking settings interactively."""

    print("\n=== RAG system: setup ===")
    print("Enter a full MongoDB connection string or leave it blank to enter")
    print("the username, password, and cluster host separately.")
    print("Example full URI:")
    print("  mongodb+srv://username:password@cluster0.example.mongodb.net/")
    print("Do not enter credentials separated by slashes.")
    print("Type 'load' to load MongoDB credentials from a JSON file.")

    while True:
        while True:
            uri = _ask("MongoDB URI (or type 'load' for a JSON credentials file)")
            if uri.lower() == "load":
                credentials_path = Path(
                    _ask(
                        "Credentials JSON path",
                        "config/mongodb_credentials.json",
                    )
                )
                try:
                    settings = Settings.from_credentials_file(credentials_path)
                    print(f"Loaded MongoDB settings from {credentials_path}.")
                    break
                except (FileNotFoundError, ValueError) as exc:
                    print(f"[ERROR] Could not load credentials JSON: {exc}")
                    continue
            if not uri:
                os.environ.pop("MONGODB_URI", None)
                os.environ["MONGODB_USERNAME"] = _ask("MongoDB username (e.g. db_user)")
                os.environ["MONGODB_PASSWORD"] = getpass.getpass("MongoDB password (hidden): ")
                os.environ["MONGODB_CLUSTER_HOST"] = _ask(
                    "MongoDB cluster host (e.g. cluster0.example.mongodb.net)"
                )
                settings = Settings.from_env()
                break
            if uri.startswith(("mongodb://", "mongodb+srv://")):
                if "<" in uri or ">" in uri:
                    print(
                        "The URI still contains a placeholder such as '<password>'. "
                        "Replace it with the real value, or leave the URI blank and "
                        "enter the credentials separately."
                    )
                    continue
                os.environ["MONGODB_URI"] = uri
                settings = Settings.from_env()
                break
            print(
                "Invalid MongoDB URI. It must start with 'mongodb://' or "
                "'mongodb+srv://'. Please try again."
            )

        print("\nTesting MongoDB Atlas connection...", flush=True)
        try:
            client = HostConnection.connect(settings.mongodb_uri)
            client.close()
            print("[OK] MongoDB Atlas credentials and connection verified.")
            break
        except Exception as exc:
            print(f"[ERROR] MongoDB connection failed: {exc}")
            retry = _ask("Retry MongoDB credentials", "yes").lower()
            if retry not in {"y", "yes"}:
                raise RuntimeError("MongoDB connection was not verified") from exc

    print("\nThe following settings have defaults.")
    print("Press Enter without typing anything to accept the value shown in the prompt.")
    print("\nDatabase and collection settings:")
    settings = replace(
        settings,
        database=_ask("Database name", settings.database),
        collection=_ask("Collection name", settings.collection),
        vector_index=_ask("Vector index name", settings.vector_index),
        embedding_model=_ask(
            "Hugging Face embedding model id",
            settings.embedding_model,
        ),
        llm_model=_ask(
            "Hugging Face LLM id",
            settings.llm_model,
        ),
    )

    pdf_path = Path(_ask("PDF to index", "data/alice_in_wonderland.pdf"))
    raw_chunk_size = _ask(
        "Manual chunk size in tokens (leave blank to use the model limit)"
    )
    while raw_chunk_size:
        try:
            manual_chunk_size = int(raw_chunk_size)
            if manual_chunk_size > 0:
                break
        except ValueError:
            pass
        raw_chunk_size = _ask("Enter a positive integer or leave blank for automatic sizing")
    else:
        manual_chunk_size = None
    padding = _ask_positive_int("Overlap in sentence units", 1)
    recreate_index = _ask(
        "Recreate an incompatible vector index",
        "no",
    ).lower() in {"y", "yes"}
    return settings, pdf_path, manual_chunk_size, padding, recreate_index


def _index_for_session(
    settings: Settings,
    pdf_path: Path,
    manual_chunk_size: int | None,
    padding: int,
    recreate_index: bool,
) -> tuple[VectorStore, object, object, int]:
    """Index the selected document and return the active store and models."""

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    print("\nLoading the embedding model and tokenizer...", flush=True)
    model_config = get_model_chunking_config(
        settings.embedding_model,
        manual_chunk_size=manual_chunk_size,
    )

    print("\nConnecting to MongoDB Atlas...", flush=True)
    client = HostConnection.connect(settings.mongodb_uri)
    store = VectorStore.from_client(client, settings.database, settings.collection)
    dimensions = Embedding.embedding_dimension(model_config.model)
    print("\nCreating or checking the MongoDB vector index...", flush=True)
    index_action = store.create_vector_index(
        settings.vector_index,
        dimensions,
        recreate=recreate_index,
    )
    print(f"Vector index action: {index_action}.", flush=True)

    print("\nReading the PDF and creating chunks...", flush=True)
    ingestion = DataIngestion()
    chapters = ingestion.read_with_outline(pdf_path)
    nlp = ingestion.get_sentences()
    chunker = Chunking()
    document_id = pdf_path.stem
    chunks = []

    for chapter in chapters:
        for chunk in chunker.chunk_text(
            chapter["text"],
            chunk_size=model_config.chunk_size,
            padding=padding,
            nlp=nlp,
            tokenizer=model_config.tokenizer,
        ):
            chunk.update(
                document_id=document_id,
                book=document_id,
                chapter=chapter["chapter"],
                chapter_title=chapter["chapter_title"],
            )
            chunks.append(chunk)

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    store.insert_chunks(chunks, replace_document_id=document_id)

    print("\nGenerating and validating embeddings...", flush=True)
    embedding = Embedding()
    embedded = embedding.perform_embedding(
        model_config.model,
        settings.embedding_model,
        None,
        store.collection,
    )
    print(
        f"Indexed {len(chunks)} chunks and updated {embedded} embeddings "
        f"({dimensions} dimensions; {model_config.chunk_size}-token limit)."
    )
    return store, model_config.model, model_config.tokenizer, dimensions


def _run_preflight(
    settings: Settings,
    store: VectorStore,
    embedding_model: object,
    dimensions: int,
    generator: TextGenerator,
) -> list[dict[str, str]]:
    """Run connection, embedding, retrieval, and generation smoke checks."""

    print("\n=== Preflight checks ===")
    print("[OK] MongoDB Atlas connection")
    preflight_results = [{"check": "MongoDB Atlas connection", "status": "PASS"}]

    sample = embedding_model.encode("preflight embedding check", normalize_embeddings=True)
    validate_embedding(sample, expected_dims=dimensions, require_normalized=True)
    print(f"[OK] Valid embedding: {dimensions} dimensions, normalized")
    preflight_results.append({"check": "Embedding validation", "status": "PASS"})

    count = store.collection.count_documents({"embedding": {"$exists": True}})
    if count == 0:
        raise RuntimeError("No embeddings were found in the collection")
    print(f"[OK] MongoDB contains {count} embeddings")
    preflight_results.append({"check": "Stored embeddings", "status": "PASS"})

    retriever = Retriever(
        store,
        embedding_model,
        settings.embedding_model,
        vector_index=settings.vector_index,
    )
    print("Waiting for Atlas to index the new vectors...", flush=True)
    search_results = store.wait_for_vector_results(
        retriever.embed_query("What is the document about?"),
        k=1,
        index_name=settings.vector_index,
    )
    print(f"[OK] Vector search: retrieved chunk {search_results[0]['chunk_id']}")
    preflight_results.append({"check": "Vector search", "status": "PASS"})

    print("\nLoading and testing the LLM...", flush=True)
    smoke_answer = generator.generate(
        "Reply with exactly one word: READY",
        max_new_tokens=8,
    )
    print("", flush=True)
    if not smoke_answer.strip():
        raise RuntimeError("The LLM did not produce an answer")
    print(f"[OK] LLM loaded and tested: {settings.llm_model}")
    preflight_results.append({"check": "LLM generation", "status": "PASS"})
    return preflight_results


def run_interactive() -> None:
    """Run the complete setup, preflight, and interactive query workflow."""

    try:
        settings, pdf_path, manual_chunk_size, padding, recreate_index = _collect_settings()
        store, embedding_model, _tokenizer, dimensions = _index_for_session(
            settings,
            pdf_path,
            manual_chunk_size,
            padding,
            recreate_index,
        )
        generator = TextGenerator(settings.llm_model)
        preflight_results = _run_preflight(
            settings,
            store,
            embedding_model,
            dimensions,
            generator,
        )

        generator.max_new_tokens = 300
        retriever = Retriever(
            store,
            embedding_model,
            settings.embedding_model,
            vector_index=settings.vector_index,
        )
        pipeline = RAGPipeline(retriever, generator, top_k=5)

        show_diagnostics = _ask(
            "Display database plots and preflight test results before queries?",
            "y",
        ).lower()
        if show_diagnostics in {"y", "yes"}:
            try:
                display_database_diagnostics(store.statistics(), preflight_results)
            except ImportError as exc:
                print(
                    "[WARNING] Diagnostics require pandas, seaborn, and matplotlib. "
                    f"Install requirements.txt to enable them: {exc}"
                )

        print("\n=== Query loop ===")
        print("Enter a question. Type 'exit', 'quit', or 'q' to stop.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in EXIT_COMMANDS:
                print("Closing the RAG system.")
                break
            if not query:
                continue

            result = pipeline.answer(query)
            print(f"\nAnswer:\n{result['answer']}")
            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"- chunk {source['chunk_id']} | score={source['score']:.4f} | "
                    f"{source['text'][:250]}..."
                )
    except (KeyboardInterrupt, EOFError):
        print("\nClosing the RAG system.")
    except Exception as exc:
        print(f"\nSetup or preflight failed: {exc}")


if __name__ == "__main__":
    run_interactive()
