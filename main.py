#!/usr/bin/env python3
"""Ponto de entrada na raiz do repositório para `python main.py`."""
from __future__ import annotations

from aethelion.cli import build_parser, main, settings_from

__all__ = ["build_parser", "settings_from", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
