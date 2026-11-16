"""Retrieve, augment, and generate a grounded answer."""

from __future__ import annotations

from typing import Any

from src.generation import ABSTENTION, format_prompt


class RAGPipeline:
    def __init__(
        self,
        retriever: Any,
        generator: Any,
        top_k: int = 5,
        min_score: float | None = None,
    ):
        """Configure a retriever, generator, top-k value, and optional score gate."""

        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        exact: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Retrieve context, build a grounded prompt, and generate an answer."""

        k = top_k or self.top_k
        results = (
            self.retriever.exact_search(query, k=k)
            if exact
            else self.retriever.vector_search(query, k=k)
        )

        if not results or (
            self.min_score is not None and results[0]["score"] < self.min_score
        ):
            return {"answer": ABSTENTION, "sources": results, "prompt": None}

        prompt = format_prompt(query, results)
        answer = self.generator.generate(prompt)

        if verbose:
            print("\nAnswer:\n", answer)
            print("\nSources:")
            for result in results:
                print(f"- chunk {result['chunk_id']} (score={result['score']:.4f})")

        return {"answer": answer, "sources": results, "prompt": prompt}
