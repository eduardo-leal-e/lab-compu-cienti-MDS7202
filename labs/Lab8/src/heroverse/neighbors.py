"""Búsqueda de vecinos más cercanos sobre una representación."""

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors


def vecinos_mas_cercanos(
    X: np.ndarray,
    nombres: list[str],
    consulta: str,
    k: int = 5,
    metrica: str = "euclidean",
) -> pl.DataFrame:
    """Devuelve los `k` personajes más cercanos a `consulta`.

    `X` tiene una fila por personaje, en el mismo orden que `nombres`; puede
    ser un arreglo de NumPy o una matriz dispersa, como la de TF-IDF. El
    resultado tiene las columnas `name` y `distancia`, ordenadas de menor a
    mayor distancia, y no incluye a la consulta. Si `consulta` no está en
    `nombres`, levanta `KeyError`.
    """
    if consulta not in nombres:
        raise KeyError(consulta)
    if k < 0:
        raise ValueError("k debe ser no negativo.")
    consulta_idx = nombres.index(consulta)
    cantidad = min(k + 1, len(nombres))
    if cantidad <= 1 or k == 0:
        return pl.DataFrame(schema={"name": pl.String, "distancia": pl.Float64})
    modelo = NearestNeighbors(n_neighbors=cantidad, metric=metrica).fit(X)
    distancias, indices = modelo.kneighbors(X[consulta_idx : consulta_idx + 1])
    filas = [
        (nombres[indice], float(distancia))
        for indice, distancia in zip(indices[0], distancias[0], strict=True)
        if indice != consulta_idx
    ][:k]
    return pl.DataFrame(filas, schema=["name", "distancia"], orient="row")
