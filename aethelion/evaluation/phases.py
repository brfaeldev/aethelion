"""Definição das 8 fases experimentais, hipóteses H1–H5, trajetórias de câmera e esquema CSV."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

F = np.float32


@dataclass
class Phase:
    key: str
    titulo: str
    hipotese: str
    duracao: float
    config: dict
    camera: str
    nota: str


PHASES: list[Phase] = [
    Phase("A", "z-pass, câmera externa", "H2",
          12.0, {"algorithm": "zpass", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "orbita", "Linha de base: câmera nunca entra em um volume de sombra."),
    Phase("B", "z-fail, câmera externa", "H2",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "orbita", "Mesma trajetória de A: isola o custo das tampas."),
    Phase("C", "z-pass, câmera interna", "H1",
          12.0, {"algorithm": "zpass", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "travessia", "Trajetória atravessa volumes: artefato esperado."),
    Phase("D", "z-fail, câmera interna", "H1",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "travessia", "Mesma trajetória de C: robustez esperada."),
    Phase("E", "stencil 3 bits sem wrap", "H3",
          12.0, {"algorithm": "zfail", "stencil_bits": 3, "stencil_wrap": False,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "grade", "Zona de alta sobreposição: saturação esperada."),
    Phase("F", "stencil 3 bits com wrap", "H3",
          12.0, {"algorithm": "zfail", "stencil_bits": 3, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 40.0, "infinite_extrude": False},
          "grade", "Mesma cena de E com INCR_WRAP/DECR_WRAP."),
    Phase("G", "profundidade de 12 bits", "H4",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 12, "extrude_len": 40.0, "infinite_extrude": False},
          "rasante", "Buffer de profundidade estreito (quantizado por fragmento), vista rasante."),
    Phase("H", "extrusão curta", "H5",
          12.0, {"algorithm": "zfail", "stencil_bits": 8, "stencil_wrap": True,
                 "depth_bits": 0, "extrude_len": 3.5, "infinite_extrude": False},
          "orbita", "Extrusão finita insuficiente vs. extrusão robusta: sombra truncada esperada."),
]

HIPOTESES = {
    "H1": "z-pass e z-fail concordam enquanto a câmera está fora dos volumes; "
          "a divergência acompanha o número de volumes que contêm a câmera.",
    "H2": "z-fail custa mais que z-pass por processar e preencher as tampas.",
    "H3": "sem operações de wrap, a saturação de um stencil estreito produz "
          "sombra errada; com wrap, o erro desaparece na mesma cena.",
    "H4": "reduzir a largura do buffer de profundidade degrada a máscara de "
          "sombra de forma distinta da degradação da visibilidade.",
    "H5": "extrusão finita insuficiente trunca o volume e reduz a revocação, "
          "enquanto a extrusão suficiente ou homogênea no infinito (P_inf, w=0) preserva o volume.",
}

CSV_FIELDS = [
    "t_wall", "fase", "hipotese", "frame", "backend", "origem_config",
    "algoritmo", "stencil_bits", "stencil_wrap", "extrude_len",
    "infinite_extrude", "depth_bits", "precisao", "largura", "altura",
    "ms_total", "fps", "ms_geometry", "ms_silhouette", "ms_stencil",
    "ms_composite", "tris_scene", "tris_scene_drawn", "tris_volume",
    "tris_volume_drawn", "frag_scene", "frag_stencil", "pixels_visible",
    "pixels_shadow", "volumes_with_camera", "stencil_saturation",
    "stencil_wrap_events", "stencil_max", "mem_buffers_kb",
    "mem_geometry_kb", "mem_total_kb", "medido", "tp", "fp", "fn", "tn",
    "precision", "recall", "f1", "error_rate",
]


def camera_for(kind: str, u: float) -> tuple[np.ndarray, np.ndarray]:
    """Trajetórias determinísticas. u in [0,1] dentro da fase."""
    tau = 2.0 * np.pi
    if kind == "orbita":
        ang = tau * u
        eye = np.array([3.8 * np.sin(ang), 2.34 + 0.16 * np.sin(2 * ang),
                        5.6 * np.cos(ang)], dtype=F)
        tgt = np.array([0.0, -0.35, 0.0], dtype=F)
    elif kind == "travessia":
        ang = tau * u
        eye = np.array([3.35 * np.sin(ang), -0.15 + 0.25 * np.cos(2 * ang),
                        3.8 * np.cos(ang) - 1.2], dtype=F)
        tgt = np.array([0.0, -0.45, -0.5], dtype=F)
    elif kind == "grade":
        eye = np.array([-3.65 + 0.35 * np.sin(tau * u),
                        -0.10 + 0.15 * np.cos(tau * u),
                        -5.55], dtype=F)
        tgt = np.array([2.80, -0.80, -5.65], dtype=F)
    elif kind == "rasante":
        eye = np.array([4.2 * np.cos(tau * u * 0.5), -1.05,
                        4.2 * np.sin(tau * u * 0.5)], dtype=F)
        tgt = np.array([0.0, -1.38, 0.0], dtype=F)
    else:
        eye = np.array([0.0, 1.0, 6.5], dtype=F)
        tgt = np.zeros(3, dtype=F)
    return eye, tgt
