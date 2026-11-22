#!/usr/bin/env python3
"""
Ponto de Entrada da Aplicação (main.py)

Script executável na raiz do repositório (`python main.py`) que inicializa
a aplicação Aethelion e delega o processamento de argumentos de linha de
comando para o módulo principal da CLI (`aethelion.cli`).
"""
from __future__ import annotations

from aethelion.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
