"""
Tema Visual, Constantes de Layout e Ciclos da UI (theme.py)

Centraliza a paleta de cores RGB, dimensões padrão de janela e painel, listas
de valores cíclicos para ajuste rápido de parâmetros (resoluções de 128x96 a 4K,
larguras de bits, modos de precisão e vistas) e o mapa de atalhos de teclado.
"""
from __future__ import annotations

from ..core.settings import PRESETS

DEFAULT_WIN_W = 1280
DEFAULT_WIN_H = 780
MIN_PANEL_W = 380
STATUS_H = 58

BG = (22, 22, 28)
PANEL_BG = (32, 32, 42)
PANEL_DIV = (52, 52, 68)
BTN_BG = (46, 48, 62)
BTN_HOVER = (64, 68, 88)
FG = (232, 232, 240)
ACC = (143, 208, 255)
OK = (120, 220, 140)
WARN = (240, 170, 70)
BAD = (232, 92, 80)

RES_CYCLE = [
    (128, 96),
    (160, 120),
    (192, 144),
    (256, 192),
    (320, 240),
    (400, 300),
    (512, 384),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]
ALGO_CYCLE = ["zfail", "zpass", "none"]
BITS_CYCLE = [8, 5, 4, 3, 2, 1]
DEPTH_CYCLE = [0, 24, 16, 12, 10]
PREC_CYCLE = ["float32", "float16", "fixed16_16", "fixed12_4"]
SHADING_CYCLE = ["gouraud", "flat", "unlit"]
TEX_RES_CYCLE = [16, 32, 64, 128]
SNAP_CYCLE = [0, 1, 2, 4]
VIEW_NAMES = ["cena", "volumes 3D + silhueta", "buffer stencil", "diferença (ray cast)"]
PRESET_NAMES = list(PRESETS.keys())

HELP = [
    ("ESPACO", "iniciar / parar experimento"),
    ("TAB", "alternar backend (CPU/GPU)"),
    ("M", "pausar/retomar luz (sob demanda)"),
    ("F11 / F", "alternar TELA CHEIA"),
    ("R / ] / [", "resolução interna (+/-)"),
    ("1 / 2 / 0", "z-pass / z-fail / sem sombra"),
    ("W / B / D", "wrap / stencil bits / z-buffer"),
    ("P / E / I", "precisão / extrusão L / P_inf"),
    ("V / G / C", "vista / ray cast / conformidade"),
    ("S / ESC", "captura PNG / sair"),
    ("mouse/setas", "arrastar=orbitar, roda=zoom"),
]

FOOTER_COMMANDS_ROW1 = [
    ("TAB", "Backend"),
    ("ESPAÇO", "Experimento"),
    ("M", "Pausar Luz"),
    ("V", "Vista"),
    ("G", "RayCast"),
    ("C", "Conformidade"),
    ("S", "Captura PNG"),
    ("F11/F", "Tela Cheia"),
    ("ESC", "Sair"),
    ("Arraste/Setas", "Câmera"),
    ("Scroll", "Zoom/Painel"),
]

FOOTER_COMMANDS_ROW2 = [
    ("1", "Z-Pass"),
    ("2", "Z-Fail"),
    ("0", "Sem Sombra"),
    ("W", "Stencil Wrap"),
    ("B", "Stencil Bits"),
    ("D", "Z-Buffer"),
    ("P", "Precisão"),
    ("E", "Extrusão L"),
    ("I", "Extrusão Inf (P_inf)"),
    ("R / [ ] / - +", "Resolução"),
]
