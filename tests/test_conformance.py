"""Testa os 24 casos da suíte de conformidade (OpenGL 1.4 stencil, top-left e recorte)."""
from __future__ import annotations

from aethelion.evaluation.conformance import TESTS, run


def test_all_conformance_cases():
    for name, fn, ref in TESTS:
        assert fn(), f"Falha no teste de conformidade {name} ({ref})"


def test_conformance_runner(tmp_path):
    res = run(outdir=str(tmp_path), verbose=False)
    assert res["passed"] == res["total"] == 24
    assert (tmp_path / "conformidade.csv").exists()
