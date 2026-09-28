"""Andamiaje del panel de la Parte 2. Renómbrenlo si quieren.

MATERIAL PROVISTO. Lo que ya está escrito acá —los imports, la configuración
de la página y el cargador de datos— es infraestructura y no se evalúa: son
las mismas líneas para cualquier panel sobre este dataset. Lo que sí se evalúa
son las cuatro secciones de más abajo.

Las secciones están en un orden que funciona, pero no es obligatorio:
reordenarlas o renombrarlas no descuenta. Lo que se corrige es que las cuatro
cosas estén y que cada gráfico explique por qué está ahí.

Para trabajar:

    uv run streamlit run mi_panel.py

Streamlit reejecuta el archivo completo cada vez que alguien mueve un control,
así que el navegador se actualiza solo al guardar.
"""

from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

RUTA_DATOS = Path(__file__).parent / "data" / "raw" / "penguins.csv"

st.set_page_config(
    page_title="Pingüinos de Palmer", page_icon="🐧", layout="wide"
)
st.title("🐧 Pingüinos del archipiélago de Palmer")


@st.cache_data
def cargar_datos() -> pl.DataFrame:
    """Lee el CSV sin modificarlo.

    `null_values=["NA"]` es necesario: el archivo viene de R, donde `NA` marca
    los faltantes. Sin ese argumento, polars lee las columnas numéricas como
    texto. Con él, los nulos quedan adentro — que es lo que queremos, porque
    este laboratorio no limpia nada.
    """
    return pl.read_csv(RUTA_DATOS, null_values=["NA"])


df = cargar_datos()

# --- 1) La tabla interactiva -----------------------------------------------
#
# Una tabla con el dataset que el lector pueda ordenar por cualquier columna y
# filtrar con al menos dos controles: uno categórico y uno de rango numérico.
# Esos mismos filtros deben afectar también a los cuatro gráficos. Los
# registros sin valor en la columna del filtro numérico quedan fuera de la
# selección filtrada. Si ningún registro cumple los filtros, muestren un aviso.
#
# Ordenar y buscar los trae `st.dataframe` de fábrica, sin programar nada.
# Filtrar no: los controles devuelven la selección y ustedes filtran el
# DataFrame antes de pasárselo a la tabla. Denle formato a las columnas, que
# `flipper_length_mm` no es un encabezado para mostrarle a un cliente.
#
#   https://docs.streamlit.io/develop/api-reference/data/st.dataframe
#   https://docs.streamlit.io/develop/api-reference/data/st.column_config
#   https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect
#   https://docs.streamlit.io/develop/api-reference/widgets/st.slider

especies_disponibles = sorted(df["species"].drop_nulls().unique().to_list())
masa_minima = int(df["body_mass_g"].drop_nulls().min())
masa_maxima = int(df["body_mass_g"].drop_nulls().max())

especies_seleccionadas = st.sidebar.multiselect(
    "Especie", options=especies_disponibles, default=especies_disponibles
)
rango_masa = st.sidebar.slider(
    "Masa corporal (g)",
    min_value=masa_minima,
    max_value=masa_maxima,
    value=(masa_minima, masa_maxima),
    step=100,
)

df_filtrado = df.filter(
    pl.col("species").is_in(especies_seleccionadas),
    pl.col("body_mass_g").is_between(
        rango_masa[0], rango_masa[1], closed="both"
    ),
)

st.subheader("Datos de pingüinos")
if df_filtrado.is_empty():
    st.warning("No hay pingüinos que cumplan los filtros seleccionados.")
else:
    st.dataframe(
        df_filtrado,
        column_config={
            "species": st.column_config.TextColumn("Especie"),
            "island": st.column_config.TextColumn("Isla"),
            "culmen_length_mm": st.column_config.NumberColumn(
                "Largo del pico (mm)", format="%.1f"
            ),
            "culmen_depth_mm": st.column_config.NumberColumn(
                "Alto del pico (mm)", format="%.1f"
            ),
            "flipper_length_mm": st.column_config.NumberColumn(
                "Largo de aleta (mm)", format="%d"
            ),
            "body_mass_g": st.column_config.NumberColumn(
                "Masa corporal (g)", format="%d"
            ),
            "sex": st.column_config.TextColumn("Sexo"),
        },
        hide_index=True,
        width="stretch",
    )


# --- 2) La calidad de los datos --------------------------------------------
#
# Un informe visible en la página, calculado sobre el CSV completo aunque se
# apliquen filtros: qué columnas tienen nulos y cuántos, cuál es el valor
# inesperado de `sex` y cómo pueden afectar esos problemas los recuentos,
# filtros o gráficos del panel.
#
# Las cifras se calculan desde `df`, no se escriben a mano: si el
# archivo cambiara, un número escrito a mano queda mintiendo.
#
#   https://docs.streamlit.io/develop/api-reference/status/st.warning

conteo_nulos = df.null_count().row(0, named=True)
nulos_por_columna = {
    columna: int(cantidad)
    for columna, cantidad in conteo_nulos.items()
    if cantidad > 0
}
puntos_en_sex = df.filter(pl.col("sex") == ".").height

st.subheader("Calidad de los datos")
if nulos_por_columna:
    st.dataframe(
        pl.DataFrame(
            {
                "Columna": list(nulos_por_columna),
                "Valores nulos": list(nulos_por_columna.values()),
            }
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.success("El CSV no contiene valores nulos.")
st.warning(f"El valor inesperado sex='.' aparece {puntos_en_sex} vez.")
st.caption(
    "Los nulos reducen los recuentos y los gráficos que requieren esas "
    "mediciones; el filtro de masa excluye filas sin masa registrada. "
    "El punto se cuenta como una categoría de sexo y puede alterar sus "
    "recuentos o filtros. Este informe siempre usa el CSV completo."
)


# --- 3) Los cuatro gráficos ------------------------------------------------
#
# Cuatro gráficos a elección, de al menos dos tipos distintos. Pueden reusar
# los de la Parte 1 o construir otros. Van a necesitar `plotly.express`:
# impórtenlo arriba, con el resto.
#
# Cada gráfico lleva, JUNTO A ÉL Y VISIBLE EN LA PÁGINA, por qué esa
# información es útil y por qué eligieron esa visualización. Un comentario en
# el código no cuenta: quien abre el panel no lee el código.
#
#   https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
#   https://docs.streamlit.io/develop/api-reference/text/st.caption
#   https://docs.streamlit.io/develop/api-reference/layout/st.columns

if not df_filtrado.is_empty():
    columna_1, columna_2 = st.columns(2)
    with columna_1:
        st.caption(
            "La relación entre aleta y masa permite explorar cómo cambian "
            "dos medidas; el color separa las especies."
        )
        fig_aleta_masa = px.scatter(
            df_filtrado,
            x="flipper_length_mm",
            y="body_mass_g",
            color="species",
            labels={
                "flipper_length_mm": "Largo de aleta (mm)",
                "body_mass_g": "Masa corporal (g)",
                "species": "Especie",
            },
        )
        st.plotly_chart(fig_aleta_masa, width="stretch")

    with columna_2:
        st.caption(
            "El histograma compara la distribución de masa entre especies; "
            "la superposición permite ver los rangos compartidos."
        )
        fig_distribucion = px.histogram(
            df_filtrado,
            x="body_mass_g",
            color="species",
            barmode="overlay",
            opacity=0.6,
            labels={
                "body_mass_g": "Masa corporal (g)",
                "species": "Especie",
                "count": "Pingüinos",
            },
        )
        st.plotly_chart(fig_distribucion, width="stretch")

    columna_3, columna_4 = st.columns(2)
    with columna_3:
        st.caption(
            "Las cajas comparan mediana, cuartiles y posibles valores "
            "extremos de masa por especie."
        )
        fig_cajas = px.box(
            df_filtrado,
            x="species",
            y="body_mass_g",
            color="species",
            labels={
                "species": "Especie",
                "body_mass_g": "Masa corporal (g)",
            },
        )
        st.plotly_chart(fig_cajas, width="stretch")

    with columna_4:
        st.caption(
            "Las barras comparan cuántos pingüinos de cada especie hay "
            "en cada isla dentro de la selección."
        )
        conteos = df_filtrado.group_by(["species", "island"]).len(
            name="pingüinos"
        )
        fig_conteos = px.bar(
            conteos,
            x="species",
            y="pingüinos",
            color="island",
            barmode="group",
            labels={
                "species": "Especie",
                "island": "Isla",
                "pingüinos": "Cantidad de pingüinos",
            },
        )
        st.plotly_chart(fig_conteos, width="stretch")


# --- 4) El tema --------------------------------------------------------------
#
# Este no se programa acá: vive en `.streamlit/config.toml`, al lado de este
# archivo. Ya existe, con las claves comentadas — descoméntenlas y decidan sus
# colores.
#
# Para comprobar que el suyo está haciendo algo: renombren el archivo,
# reinicien el panel y vean si cambia. Si no cambia, no lo configuraron.
#
#   https://docs.streamlit.io/develop/concepts/configuration/theming
