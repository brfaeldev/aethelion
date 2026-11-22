"""
Suíte de Conformidade do Instrumento (conformance.py)

Verifica através de 24 testes determinísticos se o emulador de stencil,
a regra de preenchimento top-left e o recorte homogêneo no plano próximo
reproduzem fielmente a semântica da especificação OpenGL 1.4 (Seções 2.11,
3.5 e 4.1.5), exportando o laudo de validação para `conformidade.csv`.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..core import stencil as sc
from ..core.clipping import clip_near
from ..core.raster import scan_triangle

F = np.float32


def _buf(bits=8, wrap=True, values=(0, 1, 2, 3, 4)):
    b = sc.StencilBuffer(len(values), 1, bits=bits, wrap=wrap)
    b.buf[0, :] = np.array(values, dtype=np.int32)
    return b


def _cmp_case(func, ref, mask, values, expected):
    got = sc.compare(func, ref, mask, np.array(values, dtype=np.int32))
    return bool(np.array_equal(got, np.array(expected, dtype=bool)))


# --------------------------------------------------------------------------- #
def t_func_never():
    return _cmp_case("NEVER", 2, 0xFF, [0, 2, 5], [False, False, False])


def t_func_always():
    return _cmp_case("ALWAYS", 2, 0xFF, [0, 2, 5], [True, True, True])


def t_func_less():
    return _cmp_case("LESS", 2, 0xFF, [1, 2, 3], [False, False, True])


def t_func_lequal():
    return _cmp_case("LEQUAL", 2, 0xFF, [1, 2, 3], [False, True, True])


def t_func_greater():
    return _cmp_case("GREATER", 2, 0xFF, [1, 2, 3], [True, False, False])


def t_func_gequal():
    return _cmp_case("GEQUAL", 2, 0xFF, [1, 2, 3], [True, True, False])


def t_func_equal():
    return _cmp_case("EQUAL", 2, 0xFF, [1, 2, 3], [False, True, False])


def t_func_notequal():
    return _cmp_case("NOTEQUAL", 2, 0xFF, [1, 2, 3], [True, False, True])


def t_func_mask():
    """A máscara de comparação deve ser aplicada aos DOIS operandos."""
    return _cmp_case("EQUAL", 0x13, 0x0F, [0x03, 0x23, 0x04],
                     [True, True, False])


def t_op_incr_saturates():
    out = sc.apply_op("INCR", [6, 7], 0, 7)
    return bool(np.array_equal(out, [7, 7]))


def t_op_decr_saturates():
    out = sc.apply_op("DECR", [1, 0], 0, 7)
    return bool(np.array_equal(out, [0, 0]))


def t_op_incr_wraps():
    out = sc.apply_op("INCR_WRAP", [6, 7], 0, 7)
    return bool(np.array_equal(out, [7, 0]))


def t_op_decr_wraps():
    out = sc.apply_op("DECR_WRAP", [1, 0], 0, 7)
    return bool(np.array_equal(out, [0, 7]))


def t_op_invert():
    out = sc.apply_op("INVERT", [0, 5], 0, 15)
    return bool(np.array_equal(out, [15, 10]))


def t_op_replace_ref_masked():
    out = sc.apply_op("REPLACE", [0, 0], 0x1A, 15)
    return bool(np.array_equal(out, [10, 10]))


def t_write_mask():
    """Bits fora da máscara de escrita não podem mudar."""
    b = _buf(bits=8, values=(0b1111,))
    b.func = "ALWAYS"
    b.write_mask = 0b0011
    b.write(np.array([0]), np.array([0]), np.array([True]), op_dppass="ZERO")
    return int(b.buf[0, 0]) == 0b1100


def t_order_stencil_before_depth():
    """sfail deve ser aplicado mesmo quando o teste de profundidade falharia."""
    b = _buf(bits=8, values=(5,))
    b.func = "NEVER"
    b.write(np.array([0]), np.array([0]), np.array([False]),
            op_dppass="KEEP", op_dpfail="KEEP", op_sfail="ZERO")
    return int(b.buf[0, 0]) == 0


def t_op_selection():
    """As três combinações devem selecionar operações distintas."""
    b = _buf(bits=8, values=(4, 4))
    b.func = "ALWAYS"
    b.write(np.array([0, 0]), np.array([0, 1]),
            np.array([True, False]),
            op_dppass="INCR", op_dpfail="DECR", op_sfail="ZERO")
    return int(b.buf[0, 0]) == 5 and int(b.buf[0, 1]) == 3


def t_saturation_counter():
    """A saturação precisa ser contabilizada (é um indicador do experimento)."""
    b = _buf(bits=3, wrap=False, values=(7,))
    b.func = "ALWAYS"
    b.write(np.array([0]), np.array([0]), np.array([True]), op_dppass="INCR")
    return b.saturation_events == 1 and int(b.buf[0, 0]) == 7


def t_wrap_counter():
    b = _buf(bits=3, wrap=True, values=(7,))
    b.func = "ALWAYS"
    b.write(np.array([0]), np.array([0]), np.array([True]), op_dppass="INCR_WRAP")
    return b.wrap_events == 1 and int(b.buf[0, 0]) == 0


def _coverage(tri_a, tri_b, w=32, h=32):
    cov = np.zeros((h, w), dtype=np.int32)
    for tri in (tri_a, tri_b):
        res = scan_triangle(tri[0], tri[1], w, h)
        if res is None:
            continue
        yy, xx = res[0], res[1]
        np.add.at(cov, (yy, xx), 1)
    return cov


def t_fill_no_double_cover():
    """Aresta compartilhada: nenhum pixel pode ser coberto duas vezes."""
    # Orientação consistente (ambos no mesmo sentido) e mista
    sx_a = np.array([4.0, 24.0, 4.0], dtype=F)
    sy_a = np.array([4.0, 4.0, 24.0], dtype=F)
    sx_b_same = np.array([24.0, 4.0, 24.0], dtype=F)
    sy_b_same = np.array([24.0, 24.0, 4.0], dtype=F)
    sx_b_opp = np.array([24.0, 4.0, 24.0], dtype=F)
    sy_b_opp = np.array([4.0, 24.0, 24.0], dtype=F)
    cov1 = _coverage((sx_a, sy_a), (sx_b_same, sy_b_same))
    cov2 = _coverage((sx_a, sy_a), (sx_b_opp, sy_b_opp))
    return int(cov1.max()) <= 1 and int(cov2.max()) <= 1


def t_fill_no_holes():
    """O quadrilátero formado pelos dois triângulos não pode ter furos."""
    sx_a = np.array([4.0, 24.0, 4.0], dtype=F)
    sy_a = np.array([4.0, 4.0, 24.0], dtype=F)
    sx_b_same = np.array([24.0, 4.0, 24.0], dtype=F)
    sy_b_same = np.array([24.0, 24.0, 4.0], dtype=F)
    sx_b_opp = np.array([24.0, 4.0, 24.0], dtype=F)
    sy_b_opp = np.array([4.0, 24.0, 24.0], dtype=F)
    cov1 = _coverage((sx_a, sy_a), (sx_b_same, sy_b_same))
    cov2 = _coverage((sx_a, sy_a), (sx_b_opp, sy_b_opp))
    return bool(np.all(cov1[5:23, 5:23] == 1)) and bool(np.all(cov2[5:23, 5:23] == 1))


def t_near_clip_positive_w():
    """Após o recorte, todo vértice precisa ter w > 0 e z_ndc >= -1."""
    poly = np.array([[0.0, 0.0, -1.0, -0.5],
                     [1.0, 0.0, 1.0, 2.0],
                     [0.0, 1.0, 1.0, 2.0]], dtype=F)
    out, _ = clip_near(poly, [])
    return (out is not None
            and bool(np.all(out[:, 3] > 0.0))
            and bool(np.all((out[:, 2] + out[:, 3]) >= -1e-6)))


def t_near_clip_rejects_fully_behind():
    poly = np.array([[0.0, 0.0, -1.0, -1.0],
                     [1.0, 0.0, -1.0, -2.0],
                     [0.0, 1.0, -1.0, -3.0]], dtype=F)
    out, _ = clip_near(poly, [])
    return out is None


TESTS = [
    ("GL_NEVER", t_func_never, "4.1.5 função de stencil"),
    ("GL_ALWAYS", t_func_always, "4.1.5 função de stencil"),
    ("GL_LESS", t_func_less, "4.1.5 função de stencil"),
    ("GL_LEQUAL", t_func_lequal, "4.1.5 função de stencil"),
    ("GL_GREATER", t_func_greater, "4.1.5 função de stencil"),
    ("GL_GEQUAL", t_func_gequal, "4.1.5 função de stencil"),
    ("GL_EQUAL", t_func_equal, "4.1.5 função de stencil"),
    ("GL_NOTEQUAL", t_func_notequal, "4.1.5 função de stencil"),
    ("mascara_comparacao", t_func_mask, "4.1.5 máscara de comparação"),
    ("GL_INCR_satura", t_op_incr_saturates, "4.1.5 operação saturada"),
    ("GL_DECR_satura", t_op_decr_saturates, "4.1.5 operação saturada"),
    ("GL_INCR_WRAP", t_op_incr_wraps, "OpenGL 1.4 - wrap no núcleo"),
    ("GL_DECR_WRAP", t_op_decr_wraps, "OpenGL 1.4 - wrap no núcleo"),
    ("GL_INVERT", t_op_invert, "4.1.5 operação"),
    ("GL_REPLACE_ref", t_op_replace_ref_masked, "4.1.5 operação"),
    ("mascara_escrita", t_write_mask, "4.1.5 máscara de escrita"),
    ("ordem_stencil_antes_depth", t_order_stencil_before_depth, "4.1 ordem dos testes"),
    ("selecao_sfail_dpfail_dppass", t_op_selection, "4.1.5 seleção de operação"),
    ("contador_saturacao", t_saturation_counter, "instrumentação do experimento"),
    ("contador_wrap", t_wrap_counter, "instrumentação do experimento"),
    ("preenchimento_sem_dupla_cobertura", t_fill_no_double_cover, "3.5 regra de preenchimento"),
    ("preenchimento_sem_furos", t_fill_no_holes, "3.5 regra de preenchimento"),
    ("recorte_near_w_positivo", t_near_clip_positive_w, "2.11 recorte"),
    ("recorte_near_descarta_atras", t_near_clip_rejects_fully_behind, "2.11 recorte"),
]


def run(outdir: str | None = None, verbose: bool = True) -> dict:
    rows, passed = [], 0
    for name, fn, ref in TESTS:
        try:
            ok = bool(fn())
            err = ""
        except Exception as exc:        # pragma: no cover
            ok, err = False, repr(exc)
        passed += int(ok)
        rows.append({"teste": name, "resultado": "PASS" if ok else "FAIL",
                     "referencia": ref, "erro": err})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    total = len(TESTS)
    if verbose:
        print(f"\nConformidade: {passed}/{total} aprovados "
              f"({100.0 * passed / total:.1f}%)")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "conformidade.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["teste", "resultado",
                                               "referencia", "erro"])
            wr.writeheader()
            wr.writerows(rows)
        if verbose:
            print(f"  -> {path}")
    return {"total": total, "passed": passed, "rows": rows}
