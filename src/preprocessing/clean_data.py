import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class CleanData:

    def __init__(self, df: pd.DataFrame):

        df = df.sort_values(by="date_local_int")
        start_date = pd.to_datetime(str(df["date_local_int"].head(1).values[0]), format="%Y%m%d")
        end_date = pd.to_datetime(str(df["date_local_int"].tail(1).values[0]), format="%Y%m%d")
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        dates_df = pd.DataFrame({
            "date_local_int": date_range.strftime("%Y%m%d").astype(int)
        })

        self.basic_df = dates_df.merge(df, on="date_local_int", how="left")

    @staticmethod
    def rolling_average_impute_values(df: pd.DataFrame, col: str, window_size: int = 21) -> pd.DataFrame:
        """
        Imputación de datos faltantes usando medias móviles
        Se usa para imputar los valores del consumo de gas como variables climáticas
        """

        null_dates = df[df[col].isna()]["date_local_int"].tolist()
        rolling_mean = df[col].rolling(window=window_size, min_periods=1, center=True).mean()
        df.loc[
            df["date_local_int"].isin(null_dates),
            col
        ] = rolling_mean.loc[
            df["date_local_int"].isin(null_dates)
        ]
        logger.info(
            f"Imputación de media móvil para datos faltantes en {col} "
            f"Valor de {window_size} como ventana"
        )
        return df

    @staticmethod
    def formatting_columns(df: pd.DataFrame, rename_cols: dict[str, str]) -> pd.DataFrame:
        """
        Renombre de columnas de las variables y formateo de la variable tiempo
        """

        df["date_local_int"] = pd.to_datetime(df["date_local_int"], format="%Y%m%d")
        df = df.rename(columns=rename_cols)
        logger.info(
            f"Renombrado de variables. Uso de {list(rename_cols.values())} "
            f"Uso de variable con temporal con formato fecha"
        )
        return df
