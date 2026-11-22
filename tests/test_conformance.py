"""
Testes Automatizados da Suíte de Conformidade OpenGL 1.4 (test_conformance.py)

Executa via `pytest` os 24 casos de verificação da unidade de stencil, da
regra de preenchimento exclusiva top-left e do recorte homogêneo no plano
próximo, assegurando 100% de aprovação e a correta exportação de `conformidade.csv`.
"""
from __future__ import annotations

from aethelion.evaluation.conformance import TESTS, run


def test_all_conformance_cases():
    for name, fn, ref in TESTS:
        assert fn(), f"Falha no teste de conformidade {name} ({ref})"


def test_conformance_runner(tmp_path):
    res = run(outdir=str(tmp_path), verbose=False)
    assert res["passed"] == res["total"] == 24
    assert (tmp_path / "conformidade.csv").exists()
