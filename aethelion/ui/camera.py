"""
Controlador de Câmera Orbital Interativa (camera.py)

Gerencia a posição da câmera em coordenadas esféricas (`yaw`, `pitch`, `dist`)
ao redor de um ponto-alvo 3D (`target`), processando arraste de mouse, roda
de zoom e teclas direcionais para navegação livre na cena quando o roteiro
experimental automático não está ativo.
"""
from __future__ import annotations

import numpy as np


class OrbitCamera:
    """Câmera orbital esférica em torno de um ponto-alvo (`target`)."""

    def __init__(
        self,
        yaw: float = 0.5,
        pitch: float = 0.18,
        dist: float = 7.0,
        target: tuple[float, float, float] = (0.0, 0.25, 0.0),
    ):
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.dist = float(dist)
        self.target = np.array(target, dtype=np.float32)

    @property
    def eye(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return np.array(
            [
                self.dist * cp * np.sin(self.yaw),
                self.dist * np.sin(self.pitch) + 0.8,
                self.dist * cp * np.cos(self.yaw),
            ],
            dtype=np.float32,
        )

    def apply_to(self, renderer) -> None:
        renderer.camera_eye = self.eye
        renderer.camera_target = self.target.copy()

    def orbit(self, dx: float, dy: float, sensitivity: float = 0.008) -> None:
        self.yaw += dx * sensitivity
        self.pitch = float(np.clip(self.pitch + dy * sensitivity, -1.0, 1.3))

    def zoom(self, wheel_y: float, step: float = 0.45) -> None:
        self.dist = float(np.clip(self.dist - wheel_y * step, 2.0, 22.0))

    def handle_keys(self, keys, pg) -> bool:
        """Processa setas direcionais. Retorna True se houve movimento."""
        moved = False
        if keys[pg.K_LEFT]:
            self.yaw -= 0.05
            moved = True
        if keys[pg.K_RIGHT]:
            self.yaw += 0.05
            moved = True
        if keys[pg.K_UP]:
            self.pitch = min(1.3, self.pitch + 0.04)
            moved = True
        if keys[pg.K_DOWN]:
            self.pitch = max(-1.0, self.pitch - 0.04)
            moved = True
        return moved
