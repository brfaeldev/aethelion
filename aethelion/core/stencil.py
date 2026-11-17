"""Unidade de stencil em software.

Implementa o subconjunto do estágio de stencil descrito na especificação
OpenGL 1.4 (Segal & Akeley, 2002, Seção 4.1.5): oito funções de comparação,
oito operações, máscara de comparação, máscara de escrita e largura de buffer
configurável.

Dois pontos são tratados como fatores experimentais:

* a LARGURA do buffer (1..8 bits) — contadores estreitos saturam quando muitos
  volumes de sombra se sobrepõem no mesmo pixel;
* a escolha entre operações SATURADAS (INCR/DECR) e com transbordo CIRCULAR
  (INCR_WRAP/DECR_WRAP). As variantes de wrap (originalmente EXT_stencil_wrap)
  foram promovidas ao núcleo em OpenGL 1.4 e são a condição de robustez que
  Everitt e Kilgard (2002) demonstram para shadow volumes com alta sobreposição.
"""
from __future__ import annotations

import numpy as np

FUNCS = ("NEVER", "LESS", "LEQUAL", "GREATER", "GEQUAL",
         "EQUAL", "NOTEQUAL", "ALWAYS")
OPS = ("KEEP", "ZERO", "REPLACE", "INCR", "DECR",
       "INVERT", "INCR_WRAP", "DECR_WRAP")


def compare(func: str, ref: int, mask: int, values) -> np.ndarray:
    """Teste de stencil. Semântica OpenGL: compara (ref & mask) OP (s & mask)."""
    s = np.asarray(values, dtype=np.int32) & int(mask)
    r = np.int32(int(ref) & int(mask))
    if func == "NEVER":
        return np.zeros(s.shape, dtype=bool)
    if func == "ALWAYS":
        return np.ones(s.shape, dtype=bool)
    if func == "LESS":
        return r < s
    if func == "LEQUAL":
        return r <= s
    if func == "GREATER":
        return r > s
    if func == "GEQUAL":
        return r >= s
    if func == "EQUAL":
        return r == s
    if func == "NOTEQUAL":
        return r != s
    raise ValueError(f"função de stencil desconhecida: {func}")


def apply_op(op: str, values, ref: int, max_val: int) -> np.ndarray:
    """Operação de stencil sobre um vetor de valores."""
    s = np.asarray(values, dtype=np.int32)
    if op == "KEEP":
        return s.copy()
    if op == "ZERO":
        return np.zeros_like(s)
    if op == "REPLACE":
        return np.full_like(s, int(ref) & max_val)
    if op == "INCR":
        return np.minimum(s + 1, max_val)
    if op == "DECR":
        return np.maximum(s - 1, 0)
    if op == "INVERT":
        return (~s) & max_val
    if op == "INCR_WRAP":
        return (s + 1) & max_val
    if op == "DECR_WRAP":
        return (s - 1) & max_val
    raise ValueError(f"operação de stencil desconhecida: {op}")


class StencilBuffer:
    """Buffer de stencil com largura configurável e contadores de evento."""

    def __init__(self, width: int, height: int, bits: int = 8, wrap: bool = True):
        self.width = int(width)
        self.height = int(height)
        self.set_bits(bits)
        self.wrap = bool(wrap)
        self.buf = np.zeros((self.height, self.width), dtype=np.int32)
        # Estado do estágio (default: passa sempre, como em shadow volumes)
        self.func = "ALWAYS"
        self.ref = 0
        self.func_mask = 0xFF
        self.write_mask = 0xFF
        self.reset_counters()

    # ------------------------------------------------------------------ #
    def set_bits(self, bits: int) -> None:
        self.bits = int(np.clip(int(bits), 1, 8))
        self.max_val = (1 << self.bits) - 1

    def resize(self, width: int, height: int) -> None:
        if (int(width), int(height)) != (self.width, self.height):
            self.width, self.height = int(width), int(height)
            self.buf = np.zeros((self.height, self.width), dtype=np.int32)

    def clear(self, value: int = 0) -> None:
        self.buf[:] = int(value) & self.max_val

    def reset_counters(self) -> None:
        self.fragments = 0          # fragmentos que chegaram ao estágio
        self.saturation_events = 0  # incr/decr perdidos por saturação
        self.wrap_events = 0        # transbordos circulares ocorridos

    # ------------------------------------------------------------------ #
    def incr_op(self) -> str:
        return "INCR_WRAP" if self.wrap else "INCR"

    def decr_op(self) -> str:
        return "DECR_WRAP" if self.wrap else "DECR"

    # ------------------------------------------------------------------ #
    def write(self, yy, xx, depth_pass, op_dppass: str,
              op_dpfail: str = "KEEP", op_sfail: str = "KEEP") -> None:
        """Aplica o estágio de stencil a um conjunto de fragmentos.

        A ordem segue a especificação OpenGL 1.4: o teste de stencil ocorre
        ANTES do teste de profundidade, e a operação aplicada depende das três
        combinações (sfail, dpfail, dppass).
        """
        if len(yy) == 0:
            return
        s = self.buf[yy, xx]
        self.fragments += int(s.size)

        s_pass = compare(self.func, self.ref, self.func_mask, s)
        dp = np.asarray(depth_pass, dtype=bool)

        new = s.copy()
        groups = ((~s_pass, op_sfail),
                  (s_pass & ~dp, op_dpfail),
                  (s_pass & dp, op_dppass))
        for m, op in groups:
            if not m.any() or op == "KEEP":
                continue
            sub = s[m]
            self._count(op, sub)
            new[m] = apply_op(op, sub, self.ref, self.max_val)

        wm = int(self.write_mask) & self.max_val
        self.buf[yy, xx] = (s & ~wm) | (new & wm)

    def _count(self, op: str, sub: np.ndarray) -> None:
        if op == "INCR":
            self.saturation_events += int(np.count_nonzero(sub >= self.max_val))
        elif op == "DECR":
            self.saturation_events += int(np.count_nonzero(sub <= 0))
        elif op == "INCR_WRAP":
            self.wrap_events += int(np.count_nonzero(sub >= self.max_val))
        elif op == "DECR_WRAP":
            self.wrap_events += int(np.count_nonzero(sub <= 0))
