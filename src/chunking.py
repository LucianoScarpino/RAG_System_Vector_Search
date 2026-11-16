from __future__ import annotations

from typing import Any, Optional

import spacy as sp
from tqdm.auto import tqdm

from src.ingestion import DataIngestion


class Chunking:
    """Create bounded, sentence-aware chunks for the RAG ingestion pipeline."""

    @staticmethod
    def _split_long_sentence(sentence: str, chunk_size: int) -> list[dict]:
        """Keep an exceptional long sentence within the embedding limit."""

        if len(sentence) <= chunk_size:
            return [{"text": sentence, "sentence_split": False}]

        fragments: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in sentence.split():
            if len(word) > chunk_size:
                if current:
                    fragments.append(" ".join(current))
                    current, current_len = [], 0
                fragments.extend(word[i : i + chunk_size] for i in range(0, len(word), chunk_size))
                continue

            added_len = len(word) if not current else len(word) + 1
            if current and current_len + added_len > chunk_size:
                fragments.append(" ".join(current))
                current, current_len = [word], len(word)
            else:
                current.append(word)
                current_len += added_len

        if current:
            fragments.append(" ".join(current))

        return [{"text": fragment, "sentence_split": True} for fragment in fragments]

    @staticmethod
    def _token_count(text: str, tokenizer: Any) -> int:
        """Return the number of tokens produced for one text by a tokenizer."""

        try:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        except TypeError:
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
        return len(input_ids)

    @classmethod
    def _split_long_sentence_by_tokens(
        cls,
        sentence: str,
        chunk_size: int,
        tokenizer: Any,
    ) -> list[dict]:
        """Split an oversized sentence using the selected model tokenizer."""

        if cls._token_count(sentence, tokenizer) <= chunk_size:
            return [{"text": sentence, "sentence_split": False}]

        fragments: list[str] = []
        current: list[str] = []

        for word in sentence.split():
            candidate = " ".join(current + [word])
            if current and cls._token_count(candidate, tokenizer) > chunk_size:
                fragments.append(" ".join(current))
                current = []
                candidate = word

            if cls._token_count(candidate, tokenizer) <= chunk_size:
                current.append(word)
                continue

            # Last resort for a single word that is itself too long.
            start = 0
            while start < len(word):
                end = min(len(word), start + chunk_size)
                fragment = word[start:end]
                while cls._token_count(fragment, tokenizer) > chunk_size and end > start + 1:
                    end -= 1
                    fragment = word[start:end]
                if cls._token_count(fragment, tokenizer) > chunk_size:
                    raise ValueError("chunk_size is too small for the selected tokenizer")
                fragments.append(fragment)
                start = end

        if current:
            fragments.append(" ".join(current))

        return [{"text": fragment, "sentence_split": True} for fragment in fragments]

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        padding: int = 1,
        nlp: Optional[sp.Language] = None,
        tokenizer: Any = None,
    ) -> list[dict]:
        """Split text into bounded chunks, preserving sentences where possible.

        ``padding`` is the number of sentence units repeated at a chunk
        boundary.  A sentence longer than ``chunk_size`` is split on spaces
        (and, as a last resort, characters) so it cannot exceed the embedding
        model's input budget.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")
        if not isinstance(padding, int) or isinstance(padding, bool):
            raise TypeError("padding must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if padding < 0:
            raise ValueError("padding cannot be negative")

        if nlp is None:
            nlp = sp.blank("en")
            nlp.add_pipe("sentencizer")

        units: list[dict] = []
        for sentence in nlp(text).sents:
            normalised = " ".join(str(sentence).split())
            if normalised:
                if tokenizer is None:
                    units.extend(self._split_long_sentence(normalised, chunk_size))
                else:
                    units.extend(
                        self._split_long_sentence_by_tokens(normalised, chunk_size, tokenizer)
                    )

        chunks: list[dict] = []
        start = 0
        while start < len(units):
            end = start
            current_len = 0

            while end < len(units):
                candidate = " ".join(unit["text"] for unit in units[start : end + 1])
                candidate_len = (
                    len(candidate)
                    if tokenizer is None
                    else self._token_count(candidate, tokenizer)
                )
                if end > start and candidate_len > chunk_size:
                    break
                current_len = candidate_len
                end += 1

            selected = units[start:end]
            chunk = " ".join(unit["text"] for unit in selected)
            chunks.append(
                {
                    "position": len(chunks),
                    "text": chunk,
                    "chunk_char_count": len(chunk),
                    "chunk_word_count": len(chunk.split()),
                    "chunk_token_count": (
                        len(chunk) / 4
                        if tokenizer is None
                        else self._token_count(chunk, tokenizer)
                    ),
                    "chunk_sentence_count": len(selected),
                    "sentence_split": any(unit["sentence_split"] for unit in selected),
                }
            )

            if end == len(units):
                break

            # Always advance at least one unit, even when padding is too large.
            overlap = min(padding, max(0, end - start - 1))
            start = end - overlap

        return chunks

    def process_data(
        self,
        pdf_path,
        chunk_size: int,
        padding: int,
        book_title: str,
        nlp: Optional[sp.Language] = None,
        tokenizer: Any = None,
    ):
        """Read a PDF, create chunks, and attach document-level metadata."""


        ingestion = DataIngestion()

        chapters = ingestion.read_with_outline(pdf_path)

        chunked_text = []
        for ch in tqdm(chapters, desc="Chunking chapters"):
            for chunk in self.chunk_text(
                ch["text"],
                chunk_size=chunk_size,
                padding=padding,
                nlp=nlp,
                tokenizer=tokenizer,
            ):
                chunk.update(book=book_title, chapter=ch["chapter"], chapter_title=ch["chapter_title"])
                chunked_text.append(chunk)

        # one id for the whole book (the `position` restarts at every chapter)
        for i, chunk in enumerate(chunked_text):
            chunk["chunk_id"] = i

        print("Chapters:", len(chapters))
        print("Total extracted chunks:", len(chunked_text))

        return chunked_text
