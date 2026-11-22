"""
Estruturas de Malha Fechada e Topologia Two-Manifold (mesh.py)

Define as estruturas de dados geométricas (`Solid` e `Prop`), a soldagem
espacial de vértices coincidentes (`weld`), o cálculo de normais suaves
ponderadas por área e o mapeamento de meias-arestas opostas (`topology`)
necessário para extração eficiente de silhuetas sem costuras abertas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathutil as mu

F = np.float32


def weld(positions: np.ndarray, indices: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Solda vértices coincidentes. Sem isso as costuras quebram a adjacência.

    Quantiza as coordenadas em uma grade espacial de passo `eps`, unifica vértices
    na mesma célula e descarta triângulos degenerados (com índices repetidos).
    """
    key = np.round(positions.astype(np.float64) / eps).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    newpos = positions[first].astype(F)
    newidx = inv[indices].astype(np.int32)
    ok = ((newidx[:, 0] != newidx[:, 1])
          & (newidx[:, 1] != newidx[:, 2])
          & (newidx[:, 0] != newidx[:, 2]))
    return newpos, newidx[ok]


def topology(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arestas orientadas e face oposta de cada uma.

    Retorna (dir_edges (3T,2), edge_face (3T,), edge_opp (3T,)).
    edge_opp[e] = -1 quando a aresta é de borda (malha aberta).
    Numa malha 2-manifold fechada, cada par não-ordenado {u, v} aparece
    exatamente duas vezes (como (u, v) e (v, u) em faces adjacentes).
    """
    t = int(indices.shape[0])
    de = np.concatenate([indices[:, [0, 1]], indices[:, [1, 2]],
                         indices[:, [2, 0]]], axis=0).astype(np.int32)
    face = np.concatenate([np.arange(t), np.arange(t), np.arange(t)]).astype(np.int32)
    key = np.sort(de, axis=1)
    opp = np.full(de.shape[0], -1, dtype=np.int32)
    seen: dict[tuple, int] = {}
    for e in range(de.shape[0]):
        k = (int(key[e, 0]), int(key[e, 1]))
        j = seen.pop(k, None)
        if j is None:
            seen[k] = e
        else:
            opp[e] = face[j]
            opp[j] = face[e]
    return de, face, opp


def vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Calcula normais suaves por vértice ponderadas pela área das faces adjacentes."""
    v0, v1, v2 = (positions[indices[:, i]] for i in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    acc = np.zeros_like(positions)
    for i in range(3):
        np.add.at(acc, indices[:, i], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    return (acc / np.maximum(n, 1e-9)).astype(F)


@dataclass
class Solid:
    """Malha fechada com topologia pré-computada."""

    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    dir_edges: np.ndarray
    edge_face: np.ndarray
    edge_opp: np.ndarray

    @property
    def tri_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def closed(self) -> bool:
        return bool(np.all(self.edge_opp >= 0))


def make_solid(positions, indices) -> Solid:
    """Constrói um `Solid` soldando vértices e pré-calculando normais e adjacência."""
    p, i = weld(np.asarray(positions, F), np.asarray(indices, np.int32))
    de, ef, eo = topology(i)
    return Solid(p, i, vertex_normals(p, i), de, ef, eo)


@dataclass
class Prop:
    """Instância posicionada de um sólido na cena."""

    solid: Solid
    model: np.ndarray = field(default_factory=mu.identity)
    tint: tuple = (0.75, 0.73, 0.70)
    occluder: bool = True
    texture: str = "bricks"
    name: str = "prop"
