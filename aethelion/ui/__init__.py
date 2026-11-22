"""
Subpacote da Interface Gráfica Interativa (ui/__init__.py)

Agrupa e exporta os módulos da aplicação visual em Pygame, incluindo o tema
e constantes de layout (`theme`), a câmera orbital (`OrbitCamera`), o canvas
de controles clicáveis (`UIPanelCanvas`), o painel lateral de telemetria
(`panel`) e o controlador principal da janela (`Viewer`).
"""
from . import camera, panel, theme, viewer, widgets
from .camera import OrbitCamera
from .viewer import Viewer, main
from .widgets import UIPanelCanvas

__all__ = [
    "theme",
    "camera",
    "widgets",
    "panel",
    "viewer",
    "OrbitCamera",
    "UIPanelCanvas",
    "Viewer",
    "main",
]
