"""Ask a question against the indexed MongoDB Atlas collection."""

from __future__ import annotations

import argparse
from dataclasses import replace

from src.config import HostConnection, Settings
from src.embeddings import Embedding
from src.generation import TextGenerator
from src.rag_pipeline import RAGPipeline
from src.retrieval import Retriever
from src.vector_store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for a single RAG question."""

    parser = argparse.ArgumentParser(description="Ask the RAG system a question")
    parser.add_argument("query", help="Question to answer")
    parser.add_argument("--embedding-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--exact", action="store_true", help="Use exact Python k-NN")
    parser.add_argument("--database")
    parser.add_argument("--collection")
    parser.add_argument("--vector-index")
    return parser


def ask_question(args: argparse.Namespace) -> dict:
    """Load the configured models, answer one question, and print its sources."""

    settings = Settings.from_env()
    changes = {}
    if args.embedding_model:
        changes["embedding_model"] = args.embedding_model
    if args.llm_model:
        changes["llm_model"] = args.llm_model
    if args.database:
        changes["database"] = args.database
    if args.collection:
        changes["collection"] = args.collection
    if args.vector_index:
        changes["vector_index"] = args.vector_index
    settings = replace(settings, **changes)

    client = HostConnection.connect(settings.mongodb_uri)
    store = VectorStore.from_client(client, settings.database, settings.collection)
    embedding_model = Embedding.set_embedding_model(settings.embedding_model)
    retriever = Retriever(
        store,
        embedding_model,
        settings.embedding_model,
        vector_index=settings.vector_index,
    )
    generator = TextGenerator(settings.llm_model, max_new_tokens=args.max_new_tokens)
    pipeline = RAGPipeline(
        retriever,
        generator,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    result = pipeline.answer(args.query, exact=args.exact, verbose=False)
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- chunk {source['chunk_id']} | score={source['score']:.4f} | "
            f"{source['text'][:300]}..."
        )
    return result


def main() -> None:
    """Parse command-line arguments and answer the requested question."""

    ask_question(build_parser().parse_args())


if __name__ == "__main__":
    main()
