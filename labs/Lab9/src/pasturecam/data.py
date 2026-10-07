"""Transformaciones de la tabla de mediciones de biomasa."""

import polars as pl

OBJETIVOS = (
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
)


def a_formato_ancho(df_largo: pl.DataFrame) -> pl.DataFrame:
    """Pivotea la tabla de mediciones de formato largo a ancho.

    `df_largo` trae una fila por imagen y objetivo: `target_name` nombra el
    objetivo, `target` trae su valor, `sample_id` identifica esa fila
    (imagen + objetivo) y el resto de las columnas (incluido `image_path`)
    se repite igual en las cinco filas de una misma imagen. El resultado
    trae una fila por imagen, con una columna por cada nombre de
    `OBJETIVOS` y los metadatos conservados sin duplicar.
    """
    df_ancho = df_largo.pivot(
        on="target_name",
        index=[
            col
            for col in df_largo.columns
            if col not in ("sample_id", "target_name", "target")
        ],
        values="target",
    )
    return df_ancho
