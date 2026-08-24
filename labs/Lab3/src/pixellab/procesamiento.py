"""Operaciones de procesamiento de imágenes para completar."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from src.pixellab.imagen import Imagen


class LibImagen:
    """Filtros y transformaciones que reciben y retornan ``Imagen``."""

    def to_negative(self, img_in: Imagen) -> Imagen:
        # Su código aquí
        resultado = (255 - img_in.imagen).astype(int).copy()
        return Imagen(resultado)

    def to_gray(self, img_in: Imagen) -> Imagen:
        # Su código aquí
        img = img_in.imagen
        gris = (
            0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        )
        resultado = np.stack([gris, gris, gris], axis=2).astype(int)
        return Imagen(resultado)

    def get_channel(self, img_in: Imagen, channel: str) -> Imagen:
        # Su código aquí
        canales = {"r": 0, "g": 1, "b": 2}
        if channel not in canales:
            raise ValueError(
                f"Canal '{channel}' no válido. "
                "Valores posibles: 'r', 'g' o 'b'."
            )

        resultado = np.zeros_like(img_in.imagen, dtype=int)
        indice = canales[channel]
        resultado[:, :, indice] = img_in.imagen[:, :, indice]
        return Imagen(resultado.astype(int))

    def flip(self, img_in: Imagen, axis: str) -> Imagen:
        # Su código aquí
        if axis == "h":
            resultado = img_in.imagen[:, ::-1, :].astype(int).copy()
        elif axis == "v":
            resultado = img_in.imagen[::-1, :, :].astype(int).copy()
        else:
            raise ValueError(
                f"Eje '{axis}' no válido. "
                "Valores posibles: 'h' (horizontal) o 'v' (vertical)."
            )

        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def set_saturation(self, img_in: Imagen, C: float) -> Imagen:
        # Su código aquí
        gris = self.to_gray(img_in).imagen
        resultado = (gris + C * (img_in.imagen - gris)).astype(int)
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def set_contrast(self, img_in: Imagen, C: float) -> Imagen:
        # Su código aquí
        F = 259 * (C + 255) / (255 * (259 - C))
        resultado = (F * (img_in.imagen - 128) + 128).astype(int)
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def conv_channel(self, img_in: Imagen, kernel: np.ndarray) -> Imagen:
        """Aplica una convolución 2D con el kernel sobre cada canal RGB.

        La convolución combina cada píxel con sus vecinos según los pesos
        del kernel para producir efectos como enfoque, desenfoque o relieve.
        """
        # El cuerpo de este método lo entrega el curso.
        img = img_in.imagen
        img_out = []
        for i in range(img.shape[-1]):
            img_channel = convolve2d(
                img[:, :, i], kernel, mode="same", boundary="symm"
            )
            img_out.append(img_channel)
        new_image = np.stack(img_out, axis=2)
        new_image[new_image > 255], new_image[new_image < 0] = 255, 0
        return Imagen(new_image.astype(int))
