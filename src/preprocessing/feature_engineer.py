from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()


class FeatureEngineer:

    def __init__(
            self,
            target: str = "consumo_gas",
            feats: Optional[List[str]] = None,
            lags: Optional[Dict[str, List[int]]] = None,
            windows: Optional[List[int]] = None,
            shift: int = 1
    ):
        columns_feats = ["temperatura", "precipitacion", "velocidad_viento", "numero_habitantes"]
        self.target = target
        self.feats = feats if feats is not None else columns_feats
        self.lags = lags if lags is not None else [1, 7, 15]
        self.windows = windows if windows is not None else [3, 7, 15]
        self.shift = shift

        self.columns = [self.target]
        self.columns.extend(self.feats)

    def get_lags_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtención de lags para consumo de gas, temperatura, precipitación y viento
        """
        for var in self.columns:
            lags_list = list(self.lags[var])
            logger.info(f"Lags {lags_list} a calcular para {var}")
            for lag in lags_list:
                df[f'{var}_lag_{lag}'] = df[var].shift(lag)
        logger.info(f"Feature lags incorporados para {self.columns}")
        return df

    def get_rolling_windows_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtención de lags para consumo de gas, temperatura, precipitación y viento
        La media móvil obtenida debe ser no centrada debido
        a que la usaremos para ayudarnos en la predicción (evitar data-leakage)
        """
        for var in self.columns:
            for window in self.windows:
                df[f'{var}_rolling_window_{window}'] = df[var].rolling(window, center=False).mean()
                if var == "consumo_gas":
                    df[f'{var}_rolling_window_{window}_std'] = df[var].rolling(window, center=False).std()
        logger.info(f"Feature rolling window {self.windows} incorporados para {self.columns}")
        return df

    def get_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtención de variables relacionadas con la fecha
        - día de la semana
        - quincena del mes
        - semana del año
        - indicador fin de semana
        - indicador festivo, día previo o día posterior
        """
        df = df.sort_values('fecha')
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['quincena'] = (df['fecha'].dt.day > 15).astype(int)
        df['semana_anyo'] = df['fecha'].dt.isocalendar().week

        df = self.__get_holiday_feats(df)
        df['festivo_previo'] = df['es_festivo'].shift(self.shift).fillna(0)
        df['festivo_siguiente'] = df['es_festivo'].shift(-self.shift).fillna(0)
        logger.info(
            "Obtención de feautres sobre fecha "
            "(día de la semana, quincena del mes, indicador festivo, semana del año, etc.)"
        )
        return df

    @staticmethod
    def __get_holiday_feats(df: pd.DataFrame) -> pd.DataFrame:
        df["es_festivo"] = np.where(df["numero_habitantes"].isna(), 0, 1)
        return df
