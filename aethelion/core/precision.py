"""Emulação controlada de precisão aritmética e largura de Z-buffer."""
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
    """Emula um buffer de profundidade de largura reduzida por fragmento (z em NDC [-1, 1])."""
    if bits <= 0 or bits >= 32:
        return np.asarray(z, dtype=F)
    levels = F((1 << int(bits)) - 1)
    d = np.clip((np.asarray(z, dtype=F) * F(0.5) + F(0.5)), 0.0, 1.0)
    return (np.round(d * levels) / levels * F(2.0) - F(1.0)).astype(F)
