"""CLI entry point to run the FastAPI + Gradio server for the DNI pipeline."""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

import uvicorn

from .api import app
from ..logging_service import logging_service


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the server command."""
    parser = argparse.ArgumentParser(
        description="Run the DNI extraction FastAPI server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the FastAPI/Gradio server."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging_service.configure(verbose=args.verbose)

    log_level = "debug" if args.verbose else "info"
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level)
    return 0


__all__ = ["main", "build_parser"]
