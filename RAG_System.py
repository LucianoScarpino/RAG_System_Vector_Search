from __future__ import annotations

import argparse
import sys

from scripts.ask_question import ask_question, build_parser as question_parser
from scripts.index_document import index_document, build_parser as index_parser
from scripts.interactive import run_interactive


def main() -> None:
    """Dispatch the interactive application or one of the CLI subcommands."""

    if len(sys.argv) == 1:
        run_interactive()
        return

    parser = argparse.ArgumentParser(description="MongoDB RAG system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    interactive = subparsers.add_parser(
        "interactive",
        help="configure the system and start the query loop",
    )
    interactive.set_defaults(handler=lambda _args: run_interactive())

    index = subparsers.add_parser("index", parents=[index_parser()], add_help=False)
    index.set_defaults(handler=index_document)

    ask = subparsers.add_parser("ask", parents=[question_parser()], add_help=False)
    ask.set_defaults(handler=ask_question)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
