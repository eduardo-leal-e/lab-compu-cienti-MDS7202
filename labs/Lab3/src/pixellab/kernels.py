"""Kernels de convolución que deben definir para la Etapa 6."""

import numpy as np

# Su código aquí: agreguen al menos cinco tuplas (nombre, kernel).
KERNELS: list[tuple[str, np.ndarray]] = [
    # Mantiene la imagen sin cambios al conservar solo el píxel central.
    ("identidad", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # Detecta bordes resaltando cambios bruscos entre píxeles vecinos.
    ("laplaciano", np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])),
    # Aumenta la nitidez reforzando el centro respecto de sus vecinos.
    ("enfoque", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    # Suaviza la imagen promediando cada píxel con sus ocho vecinos.
    ("desenfoque", np.ones((3, 3)) / 9),
    # Simula profundidad al iluminar y oscurecer bordes opuestos.
    ("relieve", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])),
]
