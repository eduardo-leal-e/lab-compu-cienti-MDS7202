"""Funciones para declarar y validar el esquema CRU."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

ESQUEMA_TEMPERATURAS = pa.DataFrameSchema(
    {
        "country": pa.Column(pl.String),
        "iso_alpha2": pa.Column(pl.String),
        "iso_alpha3": pa.Column(pl.String),
        "year": pa.Column(
            pl.Int64,
            checks=pa.Check.in_range(1901, 2025),
        ),
        "period": pa.Column(
            pl.String,
            checks=pa.Check.isin(
                [
                    "JAN",
                    "FEB",
                    "MAR",
                    "APR",
                    "MAY",
                    "JUN",
                    "JUL",
                    "AUG",
                    "SEP",
                    "OCT",
                    "NOV",
                    "DEC",
                    "DJF",
                    "MAM",
                    "JJA",
                    "SON",
                    "ANN",
                ]
            ),
        ),
        "temperature_c": pa.Column(
            pl.Float64,
            nullable=True,
        ),
        "parameter": pa.Column(
            pl.String,
            checks=pa.Check.equal_to("Mean Temperature"),
        ),
        "units": pa.Column(
            pl.String,
            checks=pa.Check.equal_to("degrees Celsius"),
        ),
        "source_file": pa.Column(pl.String),
    }
)


def comparar_esquema(temperaturas: pl.DataFrame) -> list[str]:
    """Devuelve diferencias entre el esquema real y el esperado."""
    esquema_esperado = {
        "country": pl.String,
        "iso_alpha2": pl.String,
        "iso_alpha3": pl.String,
        "year": pl.Int64,
        "period": pl.String,
        "temperature_c": pl.Float64,
        "parameter": pl.String,
        "units": pl.String,
        "source_file": pl.String,
    }

    diferencias = [
        columna
        for columna, tipo in esquema_esperado.items()
        if temperaturas.schema.get(columna) != tipo
    ]

    return diferencias


def validar_esquema(temperaturas: pl.DataFrame) -> None:
    """Comprueba los nombres y tipos de las columnas."""
    diferencias = comparar_esquema(temperaturas)

    if diferencias:
        raise ValueError(f"Esquema incorrecto en: {', '.join(diferencias)}")


def validar_datos(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Valida tipos, periodos, unidades y valores faltantes."""
    validar_esquema(temperaturas)

    return ESQUEMA_TEMPERATURAS.validate(temperaturas)


def casos_que_fallan(temperaturas: pl.DataFrame) -> pl.DataFrame:
    """Devuelve los incumplimientos sin ocultar sus columnas."""
    try:
        ESQUEMA_TEMPERATURAS.validate(
            temperaturas,
            lazy=True,
        )
    except pa.errors.SchemaErrors as error:
        return error.failure_cases

    return pl.DataFrame()
