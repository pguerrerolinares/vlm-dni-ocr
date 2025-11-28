"""Convenience launcher for the FastAPI + Gradio server."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from dni_pipeline.server import main as server_main  # pylint: disable=import-error

    return server_main()


if __name__ == "__main__":
    raise SystemExit(main())
