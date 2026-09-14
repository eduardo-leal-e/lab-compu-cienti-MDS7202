"""Ingesta reproducible de las fuentes Parquet hacia Bronze."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_sources(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """Lee las cuatro fuentes crudas y conserva exactamente su esquema."""

    ruta_orders = raw_dir / "orders.parquet"
    ruta_customers = raw_dir / "customers.parquet"
    ruta_order_items = raw_dir / "order_items.parquet"
    ruta_payments = raw_dir / "payments.parquet"

    if not ruta_orders.exists():
        raise FileNotFoundError("Falta la fuente orders")

    if not ruta_customers.exists():
        raise FileNotFoundError("Falta la fuente customers")

    if not ruta_order_items.exists():
        raise FileNotFoundError("Falta la fuente order_items")

    if not ruta_payments.exists():
        raise FileNotFoundError("Falta la fuente payments")

    orders = pl.read_parquet(ruta_orders)
    customers = pl.read_parquet(ruta_customers)
    order_items = pl.read_parquet(ruta_order_items)
    payments = pl.read_parquet(ruta_payments)

    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "payments": payments,
    }
