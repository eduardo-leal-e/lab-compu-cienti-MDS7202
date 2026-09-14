"""Productos analíticos Gold de Pudubella."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def _segment_expression(segments: dict[str, Any]) -> pl.Expr:
    """Construye la clasificación RFM respetando el orden de configuración."""
    expression: pl.Expr = pl.lit("Other")
    rules_by_segment = segments["segments"]
    for name, rules in reversed(list(rules_by_segment.items())):
        conditions: list[pl.Expr] = []
        if "max_recency_days" in rules:
            conditions.append(
                pl.col("recency_days") <= rules["max_recency_days"]
            )
        if "min_recency_days" in rules:
            conditions.append(
                pl.col("recency_days") >= rules["min_recency_days"]
            )
        if "min_frequency" in rules:
            conditions.append(pl.col("frequency") >= rules["min_frequency"])
        if "min_monetary" in rules:
            conditions.append(pl.col("monetary") >= rules["min_monetary"])
        condition = pl.all_horizontal(conditions)
        expression = (
            pl.when(condition).then(pl.lit(name.title())).otherwise(expression)
        )
    return expression.alias("segment")


def build_rfm_exclusions(
    orders: pl.DataFrame, payments: pl.DataFrame
) -> pl.DataFrame:
    """Registra órdenes entregadas sin pago para excluirlas de RFM."""
    paid_order_ids = payments.select("order_id").unique()
    return (
        orders.filter(pl.col("order_status") == "delivered")
        .join(paid_order_ids, on="order_id", how="anti")
        .select(
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            pl.lit("delivered_order_without_payment").alias("reason"),
        )
    )


def build_sales_daily(
    orders: pl.DataFrame, items: pl.DataFrame
) -> pl.DataFrame:
    """Construye ventas de ítems por fecha de compra y órdenes entregadas."""
    return (
        orders.filter(pl.col("order_status") == "delivered")
        .select("order_id", "order_purchase_timestamp")
        .join(items.select("order_id", "price"), on="order_id", how="inner")
        .with_columns(
            pl.col("order_purchase_timestamp").dt.date().alias("sale_date")
        )
        .group_by("sale_date")
        .agg(
            pl.col("price").sum().alias("items_sold_value"),
            pl.col("order_id").n_unique().alias("delivered_orders"),
        )
        .sort("sale_date")
    )


def build_customer_rfm(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    payments: pl.DataFrame,
    segments: dict[str, Any],
) -> pl.DataFrame:
    """Calcula RFM de compras entregadas y aplica reglas congeladas."""
    paid_by_order = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("order_payment_value")
    )
    eligible_orders = (
        orders.filter(pl.col("order_status") == "delivered")
        .join(paid_by_order, on="order_id", how="inner")
        .join(
            customers.select("customer_id", "customer_unique_id"),
            on="customer_id",
            how="inner",
        )
    )
    reference_day = eligible_orders.select(
        pl.col("order_purchase_timestamp").max().dt.date()
    ).item() + timedelta(days=1)
    return (
        eligible_orders.group_by("customer_unique_id")
        .agg(
            pl.col("order_purchase_timestamp").max().alias("last_purchase"),
            pl.col("order_id").n_unique().alias("frequency"),
            pl.col("order_payment_value").sum().alias("monetary"),
        )
        .with_columns(
            (pl.lit(reference_day) - pl.col("last_purchase").dt.date())
            .dt.total_days()
            .cast(pl.Int64)
            .alias("recency_days")
        )
        .with_columns(_segment_expression(segments))
        .select(
            "customer_unique_id",
            "last_purchase",
            "frequency",
            "monetary",
            "recency_days",
            "segment",
        )
        .sort("customer_unique_id")
    )
