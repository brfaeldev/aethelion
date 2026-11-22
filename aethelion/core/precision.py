"""
Emulador de Precisão Aritmética e Quantização de Z-Buffer (precision.py)

Simula restrições numéricas de arquiteturas gráficas clássicas nos estágios
de transformação e rasterização, suportando ponto flutuante (`float32`,
`float16`), ponto fixo (`fixed16_16`, `fixed12_4` estilo PS1/GTE) e
discretização configurável por fragmento da profundidade (`quantize_depth`).
"""
from __future__ import annotations

import numpy as np

F = np.float32


def apply_precision(arr: np.ndarray, mode: str) -> np.ndarray:
    """Reduz a precisão numérica dos dados conforme o modo arquitetural escolhido."""
    if mode == "float32":
        return arr.astype(F, copy=False)
    if mode == "float16":
        c = np.clip(arr, -65500.0, 65500.0)
        return c.astype(np.float16).astype(F)
    if mode == "fixed16_16":          # 16.16 -> 1/65536 de resolução
        c = np.clip(arr, -32767.0, 32767.0)
        return (np.round(c * 65536.0) / 65536.0).astype(F)
    if mode == "fixed12_4":           # 12.4 -> 1/16, bem grosseiro (estilo PS1)
        c = np.clip(arr, -2047.0, 2047.0)
        return (np.round(c * 16.0) / 16.0).astype(F)
    return arr.astype(F, copy=False)


def quantize_depth(z: np.ndarray, bits: int) -> np.ndarray:
    """Emula um buffer de profundidade de largura reduzida por fragmento (z em NDC [-1, 1]).

    Mapeia z_ndc de [-1, 1] para [0, 1], discretiza em (2^bits - 1) níveis inteiros
    e reconverte para [-1, 1] antes da comparação de visibilidade/stencil.
    """
    if bits <= 0 or bits >= 32:
        return np.asarray(z, dtype=F)
    levels = F((1 << int(bits)) - 1)
    d = np.clip((np.asarray(z, dtype=F) * F(0.5) + F(0.5)), 0.0, 1.0)
    return (np.round(d * levels) / levels * F(2.0) - F(1.0)).astype(F)
