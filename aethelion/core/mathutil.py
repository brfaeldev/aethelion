"""
Utilitários de Álgebra Linear e Transformações 3D (mathutil.py)

Fornece matrizes homogêneas 4x4 para transformações afins (translação, escala,
rotações), cálculo de matriz normal 3x3 (inversa-transposta), câmera `look_at`
e projeções perspectivas tanto com plano distante finito (Blinn, 1993) quanto
no infinito (`P_inf`, Everitt & Kilgard, 2002) para extrusão de sombras com `w = 0`.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def identity() -> np.ndarray:
    return np.eye(4, dtype=F)


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=F)
    m[0, 3], m[1, 3], m[2, 3] = tx, ty, tz
    return m


def scale(sx: float, sy: float | None = None, sz: float | None = None) -> np.ndarray:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    m = np.eye(4, dtype=F)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def rotate_x(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(4, dtype=F)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotate_y(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(4, dtype=F)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotate_z(ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(4, dtype=F)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def normal_matrix(model: np.ndarray) -> np.ndarray:
    """Matriz normal 3x3 (inversa-transposta) para transformar normais sob escala não-uniforme."""
    m3 = model[:3, :3]
    try:
        return np.linalg.inv(m3).T.astype(F)
    except np.linalg.LinAlgError:
        return m3.astype(F)


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Matriz de visualização (world -> view), sistema destro olhando para -Z."""
    eye = np.asarray(eye, dtype=F)
    target = np.asarray(target, dtype=F)
    up = np.asarray(up, dtype=F)

    f = normalize(target - eye)          # frente
    s = normalize(np.cross(f, up))       # direita
    u = np.cross(s, f)                   # cima real

    m = np.eye(4, dtype=F)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Transformação de perspectiva clássica (Blinn, 1993): NDC em [-1, 1]."""
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=F)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def perspective_infinite(fovy_deg: float, aspect: float, near: float,
                         eps: float = 2.4e-7) -> np.ndarray:
    """Matriz de perspectiva com plano distante no infinito P_inf (Everitt & Kilgard, 2002, Seção 3.2).

    Quando Far -> infinito, a linha Z mapeia pontos direcionais no infinito
    (w = 0, z_view < 0) para z_ndc = 1 - eps <= 1.0, impedindo que volumes de
    sombra extrudados até o infinito sejam descartados pelo plano de recorte
    distante mesmo sem extensão de hardware NV_depth_clamp.
    """
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=F)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = F(eps - 1.0)
    m[2, 3] = F((eps - 2.0) * near)
    m[3, 2] = -1.0
    return m
