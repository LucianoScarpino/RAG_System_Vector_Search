"""Database statistics and preflight visualization helpers."""

from __future__ import annotations

from typing import Any


def build_statistics_table(records: list[dict[str, Any]]):
    """Convert MongoDB chunk statistics into an indexed diagnostic table."""

    import pandas as pd

    if not records:
        raise ValueError("No chunk statistics are available")
    return pd.DataFrame(records).set_index("chunk_id").sort_index()


def display_database_diagnostics(
    records: list[dict[str, Any]],
    preflight_results: list[dict[str, str]],
) -> None:
    """Print diagnostic tables and display database and test-result plots."""

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    document_df = build_statistics_table(records)
    tests_df = pd.DataFrame(preflight_results)
    tests_df["passed"] = (tests_df["status"] == "PASS").astype(int)

    print("\n=== Database statistics ===")
    print("\nChunk table:")
    print(document_df.to_string())
    print("\nDescriptive statistics:")
    print(document_df.describe().round(2).to_string())
    print("\nPreflight test results:")
    print(tests_df[["check", "status"]].to_string(index=False))

    sns.set_style("whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    sns.histplot(data=document_df, x="chunk_token_count", bins=20, ax=axes[0])
    sns.histplot(
        data=document_df,
        x="chunk_sentence_count",
        discrete=True,
        ax=axes[1],
    )
    sns.histplot(data=document_df, x="chunk_char_count", bins=20, ax=axes[2])
    sns.histplot(data=document_df, x="chunk_word_count", bins=20, ax=axes[3])
    sns.scatterplot(
        data=document_df,
        x="chunk_sentence_count",
        y="chunk_token_count",
        ax=axes[4],
    )
    sns.barplot(data=tests_df, x="check", y="passed", ax=axes[5], color="steelblue")

    axes[0].set_title("Token count per chunk")
    axes[1].set_title("Sentence count per chunk")
    axes[2].set_title("Character count per chunk")
    axes[3].set_title("Word count per chunk")
    axes[4].set_title("Tokens vs. sentences")
    axes[5].set_title("Preflight test status")
    axes[5].set_ylabel("Pass = 1")
    axes[5].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    plt.show()
