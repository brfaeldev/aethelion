"""
Configuração e Fixtures Compartilhadas do Pytest (conftest.py)

Disponibiliza fixtures reutilizáveis para os módulos de teste, incluindo
uma configuração leve de baixa resolução (`small_settings`) e um sólido
cúbico unitário two-manifold (`unit_box`).
"""
from __future__ import annotations

import pytest

from aethelion.core.settings import SVSettings
from aethelion.scenes.primitives import box


@pytest.fixture
def small_settings() -> SVSettings:
    """Configuração leve de baixa resolução para testes rápidos de integração."""
    return SVSettings(width=96, height=72, detail=0.6)


@pytest.fixture
def unit_box():
    """Cubo unitário fechado (two-manifold) centrado na origem."""
    return box(1.0, 1.0, 1.0)
