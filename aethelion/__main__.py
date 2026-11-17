"""Permite executar o pacote via `python -m aethelion`."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
