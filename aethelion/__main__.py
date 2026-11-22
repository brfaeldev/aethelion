"""
Executor de Módulo do Pacote (aethelion/__main__.py)

Habilita a execução direta do pacote como módulo Python (`python -m aethelion`),
encaminhando o fluxo de inicialização e o código de saída para o ponto de
entrada da interface de linha de comando (`aethelion.cli.main`).
"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
