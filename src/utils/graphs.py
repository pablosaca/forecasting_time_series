import io
import base64

from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def seasonal_decompose_plot(df: pd.DataFrame, column: str, period: int = 7) -> None:
    """
    Descomposición temporal de la serie
    """
    decomposition = seasonal_decompose(df[column], model='additive', period=period)
    decomposition.plot()
    plt.show()


def plot_acf_pacf(
        series: pd.Series,
        lags: int = 40,
        title_prefix: str = "Gráficos de autocorrelación y autocorrelación parcial",
        zero: bool = False
) -> None:
    """
    Grafica la autocorrelación y autocorrelación parcial de una serie temporal usando Plotly.
    """
    # ACF
    fig_acf, ax_acf = plt.subplots()
    plot_acf(series, lags=lags, ax=ax_acf, zero=zero)
    buf_acf = io.BytesIO()
    fig_acf.savefig(buf_acf, format="png", bbox_inches='tight')
    plt.close(fig_acf)

    # PACF
    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(series, lags=lags, ax=ax_pacf, zero=zero)
    buf_pacf = io.BytesIO()
    fig_pacf.savefig(buf_pacf, format="png", bbox_inches='tight')
    plt.close(fig_pacf)

    # Convertir a base64 para plotly
    img_acf = base64.b64encode(buf_acf.getvalue()).decode()
    img_pacf = base64.b64encode(buf_pacf.getvalue()).decode()

    # Plotly Figure
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + img_acf,
            xref="paper", yref="paper",
            x=0, y=1, sizex=0.5, sizey=1, xanchor="left", yanchor="top"
        )
    )

    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + img_pacf,
            xref="paper", yref="paper",
            x=0.5, y=1, sizex=0.5, sizey=1, xanchor="left", yanchor="top"
        )
    )

    fig.update_layout(
        title=f"{title_prefix} ACF & PACF",
        width=1000,
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.show()


def plot_multiple_time_series(
        df: pd.DataFrame, date_col: str, series_cols: List[str], title: str = "Series de tiempo"
) -> None:
    """
    Grafica dos series temporales en una misma figura con dos ejes Y (izquierda y derecha).
    """
    if len(series_cols) != 2:
        raise ValueError("Solo se pueden graficar dos columnas de series temporales.")

    fig = go.Figure()

    # pintamos primera serie
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[series_cols[0]],
        mode='lines',
        name=series_cols[0]
    ))

    # pintamos primera serie
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[series_cols[1]],
        mode='lines',
        name=series_cols[1],
        yaxis='y2'  # usa el eje Y secundario
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=series_cols[0],
        yaxis2=dict(
            title=series_cols[1],
            overlaying='y',  # superposición del eje Y secundario con el principal
            side='right',
            showgrid=False
        ),
        template="plotly_white",
        hovermode="x unified",
    )

    fig.show()


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Matriz de Correlación") -> None:
    """
    """
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix, color_continuous_scale='RdBu', color_continuous_midpoint=0, title=title
        )

    # Ajustar el tamaño del gráfico
    fig.update_layout(
        width=900,  # Aumentar el tamaño del gráfico (ajustar según sea necesario)
        height=900,  # Aumentar el tamaño del gráfico (ajustar según sea necesario)
        title_font_size=20
    )
    fig.show()
