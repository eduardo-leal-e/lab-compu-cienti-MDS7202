"""Clase ``Imagen``: contenedor de imágenes sobre el que se opera con NumPy."""

from __future__ import annotations

import numpy as np


class Imagen:
    """Contenedor de imágenes RGB.

    Completen el constructor y los operadores de esta clase siguiendo el
    contrato del enunciado y los tests de ``tests/test_imagen.py``.
    """

    def __init__(self, img: np.ndarray) -> None:
        # Su código aquí
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Debes entregar un arreglo de numpy como argumento del constructor de Imagen"
            )
        if img.ndim != 3:
            raise ValueError("no se cumple 3 dimensiones")
        if img.shape[-1] != 3:
            raise ValueError("no se cumple 3 canales")
        self.imagen = img

    def __add__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        if isinstance(other, Imagen):
            if not other.imagen.shape == self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x canales) no calzan con las de la imagen original (alto x ancho x canales)"
                )
            other = other.imagen

        resultado = (self.imagen + other).astype(int).copy()
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def __radd__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        if isinstance(other, Imagen):
            if not other.imagen.shape == self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x canales) no calzan con las de la imagen original (alto x ancho x canales)"
                )
            other = other.imagen

        resultado = (other + self.imagen).astype(int).copy()
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def __sub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        if isinstance(other, Imagen):
            if not other.imagen.shape == self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x canales) no calzan con las de la imagen original (alto x ancho x canales)"
                )
            other = other.imagen

        resultado = (self.imagen - other).astype(int).copy()
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def __rsub__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        if isinstance(other, Imagen):
            if not other.imagen.shape == self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x canales) no calzan con las de la imagen original (alto x ancho x canales)"
                )
            other = other.imagen

        resultado = (other - self.imagen).astype(int).copy()
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def __mul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        if isinstance(other, Imagen):
            if not other.imagen.shape == self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x canales) no calzan con las de la imagen original (alto x ancho x canales)"
                )
            other = other.imagen

        resultado = (self.imagen * other).astype(int).copy()
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)

    def __rmul__(self, other: int | float | np.ndarray | Imagen) -> Imagen:
        # Su código aquí
        if isinstance(other, Imagen):
            if not other.imagen.shape == self.imagen.shape:
                raise ValueError(
                    "Las dimensiones de la imagen a operar (alto x ancho x canales) no calzan con las de la imagen original (alto x ancho x canales)"
                )
            other = other.imagen

        resultado = (other * self.imagen).astype(int).copy()
        resultado[resultado > 255] = 255
        resultado[resultado < 0] = 0
        return Imagen(resultado)
