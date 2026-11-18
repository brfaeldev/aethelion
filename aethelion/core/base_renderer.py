"""Classe base compartilhada pelos renderizadores de CPU e GPU."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import mathutil as mu
from . import shadow as sv
from .clipping import clip_near
from .mesh import Prop
from .overlay import render_volume_overlay
from .precision import apply_precision
from .settings import BACKEND, SVSettings, SVStats
from .stencil import StencilBuffer
from .texture import make_textures

F = np.float32


def _default_scene_builder(detail: float) -> list[Prop]:
    from ..scenes.crypt import build_crypt
    return build_crypt(detail)


def _default_light_fn(t: float) -> np.ndarray:
    from ..scenes.crypt import light_position
    return light_position(t)


class BaseRenderer:
    """Infraestrutura comum de cena, transformação de mundo, cache de volumes e memória."""

    backend_name: str = BACKEND

    def __init__(
        self,
        settings: SVSettings | None = None,
        *,
        scene_builder: Callable[[float], list[Prop]] | None = None,
        light_fn: Callable[[float], np.ndarray] | None = None,
    ):
        self.settings = settings or SVSettings()
        self._scene_builder = scene_builder or _default_scene_builder
        self._light_fn = light_fn or _default_light_fn
        self.scene: list[Prop] = self._scene_builder(self.settings.detail)
        self.textures = make_textures(self.settings.texture_res, self.settings.color_bits)
        self.stencil = StencilBuffer(
            self.settings.width,
            self.settings.height,
            self.settings.stencil_bits,
            self.settings.stencil_wrap,
        )
        self.camera_eye = np.array([0.0, 0.9, 7.2], dtype=F)
        self.camera_target = np.array([0.0, 0.2, 0.0], dtype=F)
        self.light = self._light_fn(0.0)
        self.stats = SVStats()
        # Caches por quadro/cena:
        # - _soups: posições e normais transformadas para o espaço de mundo por id(Prop)
        # - _vol_cache_*: volumes de sombra 4D extrudados (evita reconstruir se luz/cena não mudaram)
        # - _cam_vol_cache_*: contagem de volumes que contêm a câmera (variável explicativa de H1)
        self._soups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._vol_cache_key: tuple | None = None
        self._vol_cache_data: tuple | None = None
        self._cam_vol_cache_key: tuple | None = None
        self._cam_vol_cache_val: int = 0
        # Quando True, retém posições 3D por pixel visível no G-buffer para o ray casting de referência
        self.capture_gbuffer = False
        self._last_vp: np.ndarray = mu.identity()
        self.shadow_mask = np.zeros((self.settings.height, self.settings.width), bool)
        self.visible_mask = np.zeros_like(self.shadow_mask)

    def _clear_scene_caches(self) -> None:
        self._soups.clear()
        self._vol_cache_key = None
        self._vol_cache_data = None
        self._cam_vol_cache_key = None

    def rebuild_scene(self) -> None:
        """Reconstrói a malha da cena com o `detail` atual e limpa caches."""
        self.scene = self._scene_builder(self.settings.detail)
        self._clear_scene_caches()

    def _world(self, prop: Prop) -> tuple[np.ndarray, np.ndarray]:
        """Posições e normais em espaço de mundo (cache por instância).

        Normais são transformadas pela inversa-transposta 3x3 (`normal_matrix`)
        para preservar ortogonalidade sob escalas não-uniformes nos modelos.
        """
        key = id(prop)
        cached = self._soups.get(key)
        if cached is not None:
            return cached
        pos = prop.solid.positions
        ph = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=F)], axis=1)
        pw = (prop.model @ ph.T).T[:, :3].astype(F)
        nm = mu.normal_matrix(prop.model)
        nw = (nm @ prop.solid.normals.T).T
        nw = (nw / np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-9)).astype(F)
        self._soups[key] = (pw, nw)
        return pw, nw

    def _compute_memory_stats(self, st: SVStats) -> None:
        """Calcula o perfilamento de memória arquitetural e geométrica (em KB).

        Separa o custo em duas categorias:
        1. `mem_buffers_kb`: framebuffers por pixel (cor RGB, Z-buffer e stencil
           dimensionados pelas larguras em bits configuradas) + texturas procedurais;
        2. `mem_geometry_kb`: malhas da cena (posições 3D + normais 3D em float32)
           + malhas de volumes de sombra extrudados (vértices 3D finitos ou 4D homogêneos).
        """
        s = self.settings
        px = int(s.width) * int(s.height)
        color_bytes = px * 3.0 * (int(s.color_bits) / 8.0)
        depth_b = 32.0 if s.depth_bits <= 0 else float(s.depth_bits)
        depth_bytes = px * (depth_b / 8.0)
        stencil_bytes = px * (float(s.stencil_bits) / 8.0) if s.algorithm != "none" else 0.0
        tex_bytes = (
            len(self.textures) * (int(s.texture_res) ** 2) * 3.0 * (int(s.color_bits) / 8.0)
            if s.textured else 0.0
        )
        st.mem_buffers_kb = round(
            (color_bytes + depth_bytes + stencil_bytes + tex_bytes) / 1024.0, 2
        )

        # Vértices de volume ocupam 4 floats (x,y,z,w) na extrusão homogênea ou 3 floats na finita
        vol_coords = 4 if (s.infinite_extrude or s.extrude_len <= 0.0) else 3
        scene_bytes = st.tris_scene * 3 * 6 * 4
        vol_bytes = st.tris_volume * 3 * vol_coords * 4
        st.mem_geometry_kb = round((scene_bytes + vol_bytes) / 1024.0, 2)
        st.mem_total_kb = round(st.mem_buffers_kb + st.mem_geometry_kb, 2)

    def _project(self, world_tris: np.ndarray, vp: np.ndarray) -> np.ndarray:
        """(M,3,3) ou (M,3,4) mundo -> (M,3,4) clip, com precisão numérica aplicada."""
        s = self.settings
        if world_tris.shape[-1] == 4:
            ph = world_tris.reshape(-1, 4).astype(F, copy=False)
        else:
            flat = world_tris.reshape(-1, 3)
            ph = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=F)], axis=1)
        clip = (vp @ ph.T).T.astype(F)
        # Emula quantização dos registradores de transformação (float16, fixed16_16, fixed12_4)
        clip = apply_precision(clip, s.precision)
        return clip.reshape(world_tris.shape[0], 3, 4)

    def _screen(self, clip_tri: np.ndarray):
        """(3,4) clip -> sx, sy, z(NDC), invw. None se degenerado."""
        s = self.settings
        w = clip_tri[:, 3]
        if np.any(w <= 1e-5) or not np.all(np.isfinite(w)):
            return None
        ndc = (clip_tri[:, :3] / w[:, None]).astype(F)
        ndc = apply_precision(ndc, s.precision)
        if not np.all(np.isfinite(ndc)):
            return None
        # Mapeamento de NDC [-1, 1] para coordenadas de tela [0, W] x [0, H] (eixo Y invertido)
        sx = (ndc[:, 0] * F(0.5) + F(0.5)) * F(s.width)
        sy = (F(1.0) - (ndc[:, 1] * F(0.5) + F(0.5))) * F(s.height)
        # Snap de vértices em grade inteira de pixels (emula ausência de subpixel no PS1/GTE)
        if s.vertex_snap > 0:
            g = F(s.vertex_snap)
            sx = np.round(sx / g) * g
            sy = np.round(sy / g) * g
        return sx, sy, ndc[:, 2].astype(F), (1.0 / w).astype(F)

    def _clip_parts(self, clip_tri: np.ndarray, attrs: list[np.ndarray]):
        """Recorte contra o plano próximo; devolve lista de (clip(3,4), attrs)."""
        w = clip_tri[:, 3]
        z = clip_tri[:, 2]
        # Aceitação trivial: todos os vértices têm w > 0 e estão à frente do plano near (z/w >= -1)
        if np.all(w > 1e-5) and np.all((z + w) >= 0.0):
            return [(clip_tri, attrs)]
        poly, pattrs = clip_near(clip_tri, attrs)
        if poly is None:
            return []
        # Triangulação em leque (triangle fan) do polígono convexo resultante do recorte
        out = []
        for k in range(1, poly.shape[0] - 1):
            sel = [0, k, k + 1]
            out.append((poly[sel], [a[sel] for a in pattrs]))
        return out

    def _build_cached_volumes(
        self,
        vp: np.ndarray,
        use_inf: bool,
    ) -> tuple[list[np.ndarray], list[tuple], float]:
        """Extrai silhuetas e constrói volumes de sombra homogêneos 4D (com cache)."""
        s = self.settings
        zfail = (s.algorithm == "zfail")
        ext_len = s.extrude_len if not use_inf else 40.0

        ts = time.perf_counter()
        # No modo infinito (use_inf=True), a verificação de singularidade w_clip > 1e-3
        # depende da orientação da câmera, portanto `vp` entra na chave de cache.
        cache_key = (
            tuple(float(v) for v in self.light),
            s.algorithm,
            use_inf,
            float(ext_len),
            float(s.detail),
            len(self.scene),
            vp.tobytes() if use_inf else b"",
        )
        if self._vol_cache_key == cache_key and self._vol_cache_data is not None:
            batches, occluder_cache = self._vol_cache_data
        else:
            batches = []
            occluder_cache = []
            for prop in self.scene:
                if not prop.occluder:
                    continue
                pw, _ = self._world(prop)
                if use_inf:
                    # Verifica se todas as direções de extrusão (V - L, w=0) têm w_clip > 0
                    # (apontam para dentro do semiespaço frontal da câmera). Caso algum raio
                    # aponte para trás da câmera, ancora em L=500 sob P_inf para evitar
                    # divisão por w_clip <= 0 na tampa traseira sem rasterizador homogêneo 4D.
                    d_world = pw - self.light[None, :]
                    d_h = np.concatenate([d_world, np.zeros((pw.shape[0], 1), dtype=F)], axis=1)
                    w_inf_clip = (vp @ d_h.T).T[:, 3]
                    inf_ok = bool(np.all(w_inf_clip > 1e-3))
                    sides, front, back = sv.build_volume_homogeneous(
                        pw, prop.solid.indices, prop.solid.dir_edges,
                        prop.solid.edge_face, prop.solid.edge_opp,
                        self.light, length=(ext_len if inf_ok else 500.0),
                        infinite=inf_ok, caps=zfail,
                    )
                    occluder_cache.append([prop, pw, None])
                else:
                    sides, front, back = sv.build_volume_homogeneous(
                        pw, prop.solid.indices, prop.solid.dir_edges,
                        prop.solid.edge_face, prop.solid.edge_opp,
                        self.light, length=ext_len, infinite=False, caps=zfail,
                    )
                    closed_fin = (
                        np.concatenate(
                            [x[..., :3] for x in (sides, front, back) if x.shape[0]],
                            axis=0,
                        )
                        if zfail and (sides.shape[0] or front.shape[0])
                        else None
                    )
                    occluder_cache.append([prop, pw, closed_fin])

                # z-fail exige volume fechado (laterais + tampas frontal e traseira);
                # z-pass descarta as tampas e rasteriza apenas os quadriláteros laterais.
                if zfail:
                    parts = [x for x in (sides, front, back) if x.shape[0]]
                    if parts:
                        batches.append(np.concatenate(parts, axis=0))
                elif sides.shape[0]:
                    batches.append(sides)

            self._vol_cache_key = cache_key
            self._vol_cache_data = (batches, occluder_cache)

        ms_silhouette = (time.perf_counter() - ts) * 1000.0
        return batches, occluder_cache, ms_silhouette

    def _count_volumes_with_camera(
        self,
        occluder_cache: list[list],
        ext_len: float,
    ) -> tuple[int, float]:
        """Conta quantos volumes de sombra contêm a câmera (fora do cronômetro do algoritmo).

        Retorna `(count, obs_overhead)` para que o tempo de CPU gasto nesta
        verificação (variável explicativa de H1) seja descontado de `ms_total`.
        """
        t_obs0 = time.perf_counter()
        vis_len = float(ext_len if ext_len > 0 else 40.0)
        cam_key = (
            tuple(float(v) for v in self.light),
            tuple(float(v) for v in self.camera_eye),
            vis_len,
            float(self.settings.detail),
            len(self.scene),
        )
        if self._cam_vol_cache_key == cam_key:
            return self._cam_vol_cache_val, time.perf_counter() - t_obs0

        count = 0
        for entry in occluder_cache:
            prop, pw, closed3 = entry
            if closed3 is None:
                sides3, front3, back3 = sv.build_volume(
                    pw, prop.solid.indices, prop.solid.dir_edges,
                    prop.solid.edge_face, prop.solid.edge_opp,
                    self.light, vis_len, caps=True,
                )
                if sides3.shape[0] == 0 and front3.shape[0] == 0:
                    entry[2] = np.zeros((0, 3, 3), dtype=F)
                    continue
                closed3 = np.concatenate(
                    [x for x in (sides3, front3, back3) if x.shape[0]], axis=0
                )
                entry[2] = closed3
            if closed3.shape[0] > 0 and sv.point_in_volume(self.camera_eye, closed3):
                count += 1

        self._cam_vol_cache_key = cam_key
        self._cam_vol_cache_val = count
        obs_overhead = time.perf_counter() - t_obs0
        return count, obs_overhead

    def render_volume_overlay(self, base_img: np.ndarray | None = None) -> np.ndarray:
        """Gera vista didática 3D dos Volumes de Sombra e Arestas de Silhueta."""
        return render_volume_overlay(self, base_img)
