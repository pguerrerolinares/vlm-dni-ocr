"""Convenience script to run the DNI pipeline CLI."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from dni_pipeline.adapters.cli import main as cli_main  # pylint: disable=import-error

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
