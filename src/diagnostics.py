"""Database statistics and preflight visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_DIAGNOSTICS_PATH = (
    Path(__file__).resolve().parents[1] / "reports" / "database_diagnostics.png"
)


def build_statistics_table(records: list[dict[str, Any]]):
    """Convert MongoDB chunk statistics into an indexed diagnostic table."""

    import pandas as pd

    if not records:
        raise ValueError("No chunk statistics are available")
    return pd.DataFrame(records).set_index("chunk_id").sort_index()


def display_database_diagnostics(
    records: list[dict[str, Any]],
    preflight_results: list[dict[str, str]],
    output_path: str | Path = DEFAULT_DIAGNOSTICS_PATH,
) -> Path:
    """Print diagnostics, save the plot, and display database/test results."""

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    document_df = build_statistics_table(records)
    tests_df = pd.DataFrame(preflight_results)

    print("\n=== Database statistics ===")
    print("\nChunk table:")
    print(document_df.to_string())
    print("\nDescriptive statistics:")
    print(document_df.describe().round(2).to_string())
    print("\nPreflight test results:")
    print(tests_df[["check", "status"]].to_string(index=False))

    sns.set_style("whitegrid")
    figure, axes = plt.subplot_mosaic(
        [["tokens", "sentences", "characters"], ["words", "scatter", "scatter"]],
        figsize=(14, 8),
        gridspec_kw={"wspace": 0.45, "hspace": 0.75},
    )

    sns.histplot(data=document_df, x="chunk_token_count", bins=20, ax=axes["tokens"])
    sns.histplot(
        data=document_df,
        x="chunk_sentence_count",
        discrete=True,
        ax=axes["sentences"],
    )
    sns.histplot(
        data=document_df,
        x="chunk_char_count",
        bins=20,
        ax=axes["characters"],
    )
    sns.histplot(data=document_df, x="chunk_word_count", bins=20, ax=axes["words"])
    sns.scatterplot(
        data=document_df,
        x="chunk_sentence_count",
        y="chunk_token_count",
        ax=axes["scatter"],
    )

    axes["tokens"].set_title("Token count per chunk")
    axes["sentences"].set_title("Sentence count per chunk")
    axes["characters"].set_title("Character count per chunk")
    axes["words"].set_title("Word count per chunk")
    axes["scatter"].set_title("Tokens vs. sentences")
    figure.tight_layout(pad=1.8, w_pad=2.4, h_pad=2.8)
    figure.subplots_adjust(bottom=0.14)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Diagnostic plot saved to: {output_file}")
    plt.show()
    plt.close(figure)
    return output_file
