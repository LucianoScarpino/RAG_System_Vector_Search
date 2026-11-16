import spacy
import pytest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.chunking import Chunking
from src.model_limits import (
    ChunkSizeMismatchError,
    get_model_chunking_config,
    get_model_max_tokens,
)


sns.set_style("whitegrid")


class FakeTokenizer:
    model_max_length = 128

    def __call__(self, text, add_special_tokens=True, truncation=False):
        """Return deterministic whitespace tokens for isolated chunking tests."""

        tokens = text.split()
        if add_special_tokens:
            tokens = ["<bos>"] + tokens + ["<eos>"]
        return {"input_ids": tokens}


class FakeEmbeddingModel:
    max_seq_length = 128

    def __init__(self):
        """Initialize the fake embedding model metadata."""

        self.tokenizer = FakeTokenizer()


def fake_model_loader(model_name):
    """Return a deterministic model double for model-limit tests."""

    return FakeEmbeddingModel()


@pytest.fixture()
def chunker() -> Chunking:
    """Provide a chunker instance to each test."""

    return Chunking()


@pytest.fixture()
def nlp():
    """Provide the same lightweight sentence pipeline used in production."""

    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    return pipeline


def test_empty_input_returns_no_chunks(chunker, nlp):
    """Verify that empty and whitespace-only input produces no chunks."""

    assert chunker.chunk_text("", nlp=nlp) == []
    assert chunker.chunk_text("   \n\t", nlp=nlp) == []


def test_chunks_are_non_empty_and_bounded(chunker, nlp):
    """Verify that normal chunks are non-empty and respect the character limit."""

    text = "Alice entered the room. The meeting started. Everyone took notes."

    chunks = chunker.chunk_text(text, chunk_size=35, padding=0, nlp=nlp)

    assert chunks
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(chunk["chunk_char_count"] <= 35 for chunk in chunks)


def test_sentence_length_is_calculated_from_the_current_sentence(chunker, nlp):
    """Regression test for the old len(sentences) bug."""
    text = "A short sentence. This is a much longer sentence than the first one."

    chunks = chunker.chunk_text(text, chunk_size=30, padding=0, nlp=nlp)

    assert chunks[0]["text"] == "A short sentence."
    assert chunks[1]["text"] == "This is a much longer sentence"
    assert chunks[1]["chunk_char_count"] <= 30


def test_padding_repeats_the_boundary_sentence(chunker, nlp):
    """Verify that sentence overlap is preserved across chunk boundaries."""

    text = "First sentence. Second sentence. Third sentence."

    chunks = chunker.chunk_text(text, chunk_size=35, padding=1, nlp=nlp)

    assert [chunk["text"] for chunk in chunks] == [
        "First sentence. Second sentence.",
        "Second sentence. Third sentence.",
    ]


def test_large_padding_does_not_loop_forever(chunker, nlp):
    """Verify that excessive padding still makes forward progress."""

    chunks = chunker.chunk_text("One. Two. Three. Four.", chunk_size=8, padding=100, nlp=nlp)

    assert chunks
    assert len(chunks) < 10
    assert all(chunk["text"] for chunk in chunks)


def test_long_sentence_is_bounded_and_marked(chunker, nlp):
    """Verify that oversized sentences are split and marked in metadata."""

    text = ("technical " * 20).strip() + "."

    chunks = chunker.chunk_text(text, chunk_size=30, padding=0, nlp=nlp)

    assert len(chunks) > 1
    assert all(chunk["chunk_char_count"] <= 30 for chunk in chunks)
    assert all(chunk["sentence_split"] is True for chunk in chunks)


def test_long_word_is_bounded_as_last_resort(chunker, nlp):
    """Verify that an unusually long word cannot exceed the configured limit."""

    text = "prefix " + ("x" * 80) + "."

    chunks = chunker.chunk_text(text, chunk_size=20, padding=0, nlp=nlp)

    assert all(chunk["chunk_char_count"] <= 20 for chunk in chunks)


def test_whitespace_is_normalised(chunker, nlp):
    """Verify that chunk text has stable whitespace formatting."""

    text = "  Alice   entered\n the room.\n\nShe sat down.  "

    chunks = chunker.chunk_text(text, chunk_size=100, padding=0, nlp=nlp)

    assert chunks[0]["text"] == "Alice entered the room. She sat down."
    assert "\n" not in chunks[0]["text"]
    assert "  " not in chunks[0]["text"]


def test_metadata_and_positions_are_stable(chunker, nlp):
    """Verify required metadata fields and sequential chunk positions."""

    chunks = chunker.chunk_text("A. B. C. D.", chunk_size=5, padding=0, nlp=nlp)

    expected_keys = {
        "position",
        "text",
        "chunk_char_count",
        "chunk_word_count",
        "chunk_token_count",
        "chunk_sentence_count",
        "sentence_split",
    }
    assert [chunk["position"] for chunk in chunks] == list(range(len(chunks)))
    assert all(expected_keys <= chunk.keys() for chunk in chunks)
    assert all(chunk["chunk_word_count"] == len(chunk["text"].split()) for chunk in chunks)


def test_same_input_is_deterministic(chunker, nlp):
    """Verify that identical input and configuration produce identical output."""

    text = "A reliable pipeline is repeatable. This matters in production."

    first = chunker.chunk_text(text, chunk_size=30, padding=1, nlp=nlp)
    second = chunker.chunk_text(text, chunk_size=30, padding=1, nlp=nlp)

    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"chunk_size": 0}, ValueError),
        ({"chunk_size": -1}, ValueError),
        ({"padding": -1}, ValueError),
        ({"chunk_size": 10.5}, TypeError),
        ({"padding": 1.5}, TypeError),
    ],
)
def test_invalid_configuration_fails_fast(chunker, nlp, kwargs, error):
    """Verify that invalid chunking configuration is rejected immediately."""

    with pytest.raises(error):
        chunker.chunk_text("Some text.", nlp=nlp, **kwargs)


def test_default_pipeline_is_available(chunker):
    """Verify that chunking can create its default sentence pipeline."""

    chunks = chunker.chunk_text("A sentence. Another sentence.", chunk_size=100, padding=0)

    assert len(chunks) == 1


def test_model_name_resolves_maximum_tokens():
    """Verify that a model name resolves to its effective token limit."""

    assert get_model_max_tokens("fake/model", model_loader=fake_model_loader) == 128


def test_model_limit_is_used_automatically_by_token_chunking(chunker, nlp):
    """Verify that token-aware chunking never exceeds the selected model limit."""

    config = get_model_chunking_config("fake/model", model_loader=fake_model_loader)
    text = " ".join(["word"] * 40)

    chunks = chunker.chunk_text(
        text,
        chunk_size=config.chunk_size,
        padding=0,
        nlp=nlp,
        tokenizer=config.tokenizer,
    )

    assert config.chunk_size == config.model_max_tokens == 128
    assert all(chunk["chunk_token_count"] <= config.chunk_size for chunk in chunks)


def test_matching_manual_limit_does_not_prompt():
    """Verify that an aligned manual limit does not request confirmation."""

    def fail_if_called(_):
        """Fail the test if confirmation is requested unexpectedly."""

        raise AssertionError("confirmation should not be requested")

    config = get_model_chunking_config(
        "fake/model",
        manual_chunk_size=128,
        confirm_fn=fail_if_called,
        model_loader=fake_model_loader,
    )

    assert config.chunk_size == 128


def test_mismatching_manual_limit_can_be_rejected():
    """Verify that rejecting a mismatched manual limit stops configuration."""

    config_prompt = []

    def reject(prompt):
        """Record the warning and reject the requested override."""

        config_prompt.append(prompt)
        return "n"

    with pytest.raises(ChunkSizeMismatchError, match="Chunking stopped"):
        get_model_chunking_config(
            "fake/model",
            manual_chunk_size=64,
            confirm_fn=reject,
            model_loader=fake_model_loader,
        )

    assert config_prompt
    assert "128" in config_prompt[0]
    assert "64" in config_prompt[0]


def test_mismatching_manual_limit_can_be_accepted():
    """Verify that an explicit confirmation accepts a mismatched limit."""

    config = get_model_chunking_config(
        "fake/model",
        manual_chunk_size=64,
        confirm_fn=lambda _: "y",
        model_loader=fake_model_loader,
    )

    assert config.model_max_tokens == 128
    assert config.chunk_size == 64


def chunking_statistics(chunks: list[dict], chunk_size: int | None = None) -> pd.DataFrame:
    """Create a detailed table used to inspect chunking quality."""

    return pd.DataFrame(chunks).set_index("position")


def _add_quality_columns(document_df: pd.DataFrame, chunk_size: int | None) -> pd.DataFrame:
    """Add diagnostic columns without changing the chunk objects themselves."""

    report = document_df.copy()
    report["char_per_word"] = report["chunk_char_count"] / report["chunk_word_count"]
    if chunk_size is not None:
        report["size_utilization"] = report["chunk_char_count"] / chunk_size
    return report


def plot_chunking_statistics(
    chunks: list[dict],
    chunk_size: int | None = None,
    show: bool = False,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Return the detailed chunk table and a six-panel diagnostic report."""

    document_df = _add_quality_columns(chunking_statistics(chunks, chunk_size), chunk_size)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    axes = ax.ravel()

    sns.histplot(data=document_df, ax=axes[0], x="chunk_token_count", bins=20)
    sns.histplot(data=document_df, ax=axes[1], x="chunk_sentence_count", discrete=True)

    metric_df = document_df[
        ["chunk_char_count", "chunk_word_count", "chunk_token_count"]
    ].melt(var_name="metric", value_name="value")
    sns.boxplot(data=metric_df, x="metric", y="value", ax=axes[2])

    sns.scatterplot(
        data=document_df,
        x="chunk_sentence_count",
        y="chunk_token_count",
        hue="sentence_split",
        style="sentence_split",
        ax=axes[3],
    )
    sns.lineplot(data=document_df, x=document_df.index, y="chunk_token_count", marker="o", ax=axes[4])
    sns.barplot(data=document_df, x=document_df.index, y="chunk_char_count", ax=axes[5], color="steelblue")

    axes[0].set_title("Estimated tokens distribution")
    axes[0].set_xlabel("Estimated token count")
    axes[1].set_title("Sentences per chunk distribution")
    axes[1].set_xlabel("Sentence count")
    axes[2].set_title("Chunk size dispersion")
    axes[3].set_title("Estimated tokens vs. sentences")
    axes[4].set_title("Estimated tokens by position")
    axes[4].set_xlabel("Chunk position")
    axes[5].set_title("Characters by position")
    axes[5].set_xlabel("Chunk position")
    for axis in (axes[2], axes[5]):
        axis.tick_params(axis="x", rotation=25)

    if chunk_size is not None:
        axes[5].axhline(chunk_size, linestyle="--", color="crimson", label="Character limit")
        axes[5].legend()

    fig.tight_layout()

    if show:
        print("Chunk table:\n")
        print(document_df.to_string())
        print("\nDescriptive statistics:\n")
        print(
            document_df[
                [
                    "chunk_char_count",
                    "chunk_word_count",
                    "chunk_token_count",
                    "chunk_sentence_count",
                    "char_per_word",
                ]
            ]
            .describe()
            .round(2)
        )
        plt.show()

    return document_df, fig


def test_chunking_statistics_table_and_plot(chunker, nlp):
    """Verify that the diagnostic table and six-panel report are created."""

    text = (
        "Alice entered the room. The meeting started. Everyone took notes. "
        "The discussion continued for several minutes."
    )
    chunks = chunker.chunk_text(text, chunk_size=50, padding=1, nlp=nlp)

    document_df, figure = plot_chunking_statistics(chunks, chunk_size=50)

    assert not document_df.empty
    assert "chunk_token_count" in document_df.columns
    assert "chunk_sentence_count" in document_df.columns
    assert "char_per_word" in document_df.columns
    assert "size_utilization" in document_df.columns
    assert len(figure.axes) == 6
    plt.close(figure)


def test_statistics_are_consistent_with_chunk_content(chunker, nlp):
    """Verify that reported counts match the actual chunk text."""

    chunks = chunker.chunk_text(
        "A short sentence. Another short sentence. A final sentence.",
        chunk_size=40,
        padding=0,
        nlp=nlp,
    )

    document_df = chunking_statistics(chunks, chunk_size=40)

    assert (document_df["chunk_char_count"] == document_df["text"].str.len()).all()
    assert (document_df["chunk_word_count"] == document_df["text"].str.split().str.len()).all()
    assert (document_df["chunk_token_count"] == document_df["chunk_char_count"] / 4).all()
    assert (document_df["chunk_char_count"] <= 40).all()


def test_statistics_report_reveals_oversize_sentence_chunks(chunker, nlp):
    """Verify that the report exposes sentence-splitting and size utilization."""

    text = ("unusuallylong " * 20).strip() + "."
    chunks = chunker.chunk_text(text, chunk_size=30, padding=0, nlp=nlp)

    document_df, figure = plot_chunking_statistics(chunks, chunk_size=30)

    assert document_df["sentence_split"].all()
    assert (document_df["size_utilization"] <= 1).all()
    plt.close(figure)


if __name__ == "__main__":
    demo_nlp = spacy.blank("en")
    demo_nlp.add_pipe("sentencizer")
    demo_chunks = Chunking().chunk_text(
        "Alice entered the room. The meeting started. Everyone took notes.",
        chunk_size=50,
        padding=1,
        nlp=demo_nlp,
    )
    plot_chunking_statistics(demo_chunks, chunk_size=50, show=True)
