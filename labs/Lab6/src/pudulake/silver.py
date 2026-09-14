"""Transformaciones y reglas críticas de las entidades Silver."""

from __future__ import annotations

import polars as pl

from src.pudulake.contracts import ContractViolation

DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)
ALLOWED_ORDER_STATUSES = (
    "approved",
    "canceled",
    "created",
    "delivered",
    "invoiced",
    "processing",
    "shipped",
    "unavailable",
)


def _has_rows(frame: pl.DataFrame, condition: pl.Expr) -> bool:
    """Indica si una expresión booleana selecciona al menos una fila."""
    return frame.filter(condition).height > 0


def build_orders(orders: pl.DataFrame) -> pl.DataFrame:
    """Tipa fechas de órdenes y comprueba su secuencia temporal."""
    has_unparseable_date = (
        orders.select(
            pl.any_horizontal(
                [
                    pl.col(column).is_not_null()
                    & pl.col(column)
                    .cast(pl.String)
                    .str.strptime(
                        pl.Datetime,
                        format="%Y-%m-%d %H:%M:%S",
                        strict=False,
                    )
                    .is_null()
                    for column in DATE_COLUMNS
                ]
            )
        )
        .to_series()
        .any()
    )
    if has_unparseable_date:
        raise ContractViolation("orders contiene una fecha no interpretable.")

    invalid_status = ~pl.col("order_status").is_in(
        ALLOWED_ORDER_STATUSES
    ).fill_null(False)
    if _has_rows(orders, invalid_status):
        raise ContractViolation("orders contiene un estado fuera del contrato.")

    result = orders.with_columns(
        [
            pl.col(column)
            .cast(pl.String)
            .str.strptime(
                pl.Datetime,
                format="%Y-%m-%d %H:%M:%S",
                strict=False,
            )
            for column in DATE_COLUMNS
        ]
    )
    delivered_before_purchase = (
        (pl.col("order_status") == "delivered")
        & pl.col("order_delivered_customer_date").is_not_null()
        & (
            pl.col("order_delivered_customer_date")
            < pl.col("order_purchase_timestamp")
        )
    )
    if _has_rows(result, delivered_before_purchase):
        raise ContractViolation(
            "orders tiene una entrega anterior a la compra."
        )

    return result.with_columns(
        (
            (pl.col("order_status") == "delivered")
            & pl.col("order_delivered_customer_date").is_null()
        )
        .fill_null(False)
        .alias("delivery_timestamp_missing")
    )


def build_customers(customers: pl.DataFrame) -> pl.DataFrame:
    """Conserva clientes y verifica la relación uno a uno con customer_id."""
    return customers


def build_order_items(items: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los ítems no tengan precios ni fletes negativos."""
    amount_columns = ("price", "freight_value")
    if _has_rows(
        items,
        pl.any_horizontal([pl.col(column) < 0 for column in amount_columns]),
    ):
        raise ContractViolation("order_items contiene montos negativos.")
    if _has_rows(
        items,
        pl.any_horizontal(
            [~pl.col(column).is_finite() for column in amount_columns]
        ),
    ):
        raise ContractViolation("order_items contiene montos no finitos.")
    return items


def build_payments(payments: pl.DataFrame) -> pl.DataFrame:
    """Comprueba que los pagos no tengan montos negativos."""
    if _has_rows(payments, pl.col("payment_value") < 0):
        raise ContractViolation("payments contiene montos negativos.")
    if _has_rows(payments, ~pl.col("payment_value").is_finite()):
        raise ContractViolation("payments contiene montos no finitos.")
    return payments


def validate_relationships(
    orders: pl.DataFrame,
    customers: pl.DataFrame,
    items: pl.DataFrame,
    payments: pl.DataFrame,
) -> None:
    """Verifica las claves foráneas antes de construir productos Gold."""
    relationships = (
        (
            "orders.customer_id -> customers.customer_id",
            orders,
            customers,
            "customer_id",
        ),
        (
            "order_items.order_id -> orders.order_id",
            items,
            orders,
            "order_id",
        ),
        (
            "payments.order_id -> orders.order_id",
            payments,
            orders,
            "order_id",
        ),
    )
    for name, child, parent, key in relationships:
        orphans = child.join(parent.select(key).unique(), on=key, how="anti")
        if orphans.height > 0:
            raise ContractViolation(f"Relación huérfana: {name}.")
