from typing import Optional, List, Dict, Tuple, Union, Sequence
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import STL

from src.utils.logger import get_logger

logger = get_logger()


class ConsumoGasForecaster(ABC):

    def __init__(
            self,
            df: pd.DataFrame,
            n_days_to_sequential_validation: int = 30,
            target: str = "consumo_gas",
            seed: int = 123
    ):
        self.df_original = df.copy()  # DataFrame permanente (usa todo el histórico)
        self.df = df.copy()  # DataFrame que permite ser modificado
        self.n_days_to_sequential_validation = n_days_to_sequential_validation

        self.target = target
        self.seed = seed

        self.model = None  # se define una vez entrenado el modelo

    @abstractmethod
    def sequential_train_and_forecast(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Entrenamiento y predicción del modelo de consumo de gas a nivel nacional (diario)
        El modelo se entrena y predice en cada paso, avanzando en el tiempo de forma secuencial
        """
        pass

    def reset_df(self):
        """
        Restaura el DataFrame interno a su estado original
        """
        self.df = self.df_original.copy()
        logger.info("DataFrame restaurado al estado original.")
        return self

    def filter_df_by_date(self, start_date=Optional[str], end_date=Optional[str]):
        """
        Filtra self.df por fecha y actualiza el DataFrame interno de la instancia
        """
        if start_date is not None:
            self.df = self.df[self.df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            self.df = self.df[self.df.index <= pd.to_datetime(end_date)]
        logger.info(f"DataFrame filtrado. Nuevas filas: {self.df.shape[0]}")
        return self

    @staticmethod
    def _get__metrics(
            y_pred: Union[float, int, Sequence[float], np.ndarray],
            y_test: Union[float, int, Sequence[float], np.ndarray],
    ) -> Tuple[float, float]:
        """
        Calcula métricas de bondad de ajuste (mae y mape)
        """
        y_test_arr = np.atleast_1d(y_test)
        y_pred_arr = np.atleast_1d(y_pred)
        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        mape = mean_absolute_percentage_error(y_test_arr, y_pred_arr)
        logger.info("Métricas de bondad de ajuste son calculadas (mae y mape)")
        return mae, mape

    def _get_naive_predictions(self, fecha_val, y_test: pd.Series) -> Tuple[float, float, float]:
        """
        Realización de predicciones con enfoque ingenuo: consumo del día anterior
        """
        fecha_ayer = fecha_val - pd.Timedelta(days=1)
        y_test_value = y_test.iloc[0] if isinstance(y_test, pd.Series) else y_test

        if fecha_ayer in self.df.index:
            y_naive = self.df.loc[fecha_ayer, self.target]
            mae_naive = abs(y_test_value - y_naive)
            mape_naive = abs(y_test_value - y_naive) / y_test_value
        else:
            y_naive = np.nan
            mae_naive = np.nan
            mape_naive = np.nan
        return y_naive, mae_naive, mape_naive

    @staticmethod
    def _get_reporting_data(
            output_model: List,
            importance_features: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Salida para análisis final
        """
        logger.info("Resultados finales")
        return pd.DataFrame(output_model), pd.DataFrame(importance_features).T.mean().sort_values()


class LGBMGasForecaster(ConsumoGasForecaster):

    def __init__(
            self,
            df: pd.DataFrame,
            n_days_to_sequential_validation: int = 30,
            target: str = "consumo_gas",
            categorical_features: Optional[List[str]] = None,
            lightgbm_params: Optional[Dict[str, int]] = None,
            seed: int = 123
    ):
        super().__init__(df, n_days_to_sequential_validation, target, seed)
        self.features = [col for col in df.columns if col not in ['consumo_gas']]
        aux_cat_feats = ["es_festivo", "dia_semana", "festivo_previo", "festivo_siguiente", "semana_anyo"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

        model_params = {"n_estimators": 850, "learning_rate": 0.03}
        self.lightgbm_params = lightgbm_params if lightgbm_params is not None else model_params

        if self.n_days_to_sequential_validation < 1:
            msg = (
                "El valor de n_days_to_sequential_validation debe ser mayor que cero."
                f"Se ha proporcionado un valor de {self.n_days_to_sequential_validation}, se necesita revisar."
            )
            logger.info(msg)
            raise ValueError(msg)

        self.model = None  # se actualiza cuando se entrena el modelo

    def sequential_train_and_forecast(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Entrenamiento y predicción del modelo de consumo de gas a nivel nacional (diario)
        El modelo se entrena y predice en cada paso, avanzando en el tiempo de forma secuencial
        """

        fechas_validacion = self.df.index[-self.n_days_to_sequential_validation:]

        output_model = []
        importance_features = {}
        for fecha_val in fechas_validacion:
            logger.info(
                f"Entrenamiento del modelo hasta {fecha_val}. Análisis de bondad de ajuste a fecha {fecha_val}"
            )
            train_data = self.df[self.df.index < fecha_val]
            test_data = self.df[self.df.index == fecha_val]

            if test_data.empty:
                continue

            X_train = train_data[self.features].copy()
            y_train = train_data[self.target]
            X_test = test_data[self.features].copy()
            y_test = test_data[self.target]

            X_train = self.select_categorical_features(X_train)
            X_test = self.select_categorical_features(X_test)

            self.__model_fitted(X_train, y_train)
            y_pred = self.model.predict(X_test)
            logger.info("Realización de las predicciones del modelo")
            mae, mape = self._get__metrics(y_test, y_pred)

            # obtención naive model
            y_naive, mae_naive, mape_naive = self._get_naive_predictions(fecha_val, y_test)

            output_model.append({
                'fecha_predicha': fecha_val,
                'y': y_test.iloc[0],
                'y_hat': y_pred[0],
                'y_hat_naive': y_naive,
                'mae': mae,
                'mape': mape,
                'mae_naive': mae_naive,
                'mape_naive': mape_naive
            })
            logger.info("Obtención de resultados: métricas y predicciones")
            importance_features[fecha_val] = pd.Series(self.model.feature_importances_, index=self.features)
            logger.info("Obtención variables más importantes en cada modelo")
        return self._get_reporting_data(output_model, importance_features)

    def __model_fitted(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrenamiento de un modelo LGBM a partir de la muestra de entrenamiento (feats and target column)
        """
        self.model = LGBMRegressor(
            n_estimators=self.lightgbm_params["n_estimators"],
            learning_rate=self.lightgbm_params["learning_rate"],
            random_state=self.seed,
            force_col_wise=True,
        )
        self.model.fit(
            X_train, y_train,
            categorical_feature=self.categorical_features
        )
        logger.info(
            "Modelo lightgbm ha sido entrenado "
            f'Número de estimadores {self.lightgbm_params["n_estimators"]} '
            f'y tasa de aprendizaje {self.lightgbm_params["learning_rate"]}'
        )

    def select_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Se tienen en cuenta las variables categóricas
        """
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
                logger.info(f"La variable {col} es convertida a categórica")
        return df


class STLForecaster(ConsumoGasForecaster):

    def __init__(
            self,
            df: pd.DataFrame,
            n_days_to_sequential_validation: int = 30,
            target: str = "consumo_gas",
            seed: int = 123,
            decomposition_period: int = 7
    ):
        super().__init__(df, n_days_to_sequential_validation, target, seed)
        self.period = decomposition_period

        if self.n_days_to_sequential_validation < 2 * self.period:
            msg = (
                f"El valor de n_days_to_sequential_validation debe ser mayor que el doble de {self.period}"
                f"Se ha proporcionado un valor de {self.n_days_to_sequential_validation}, se necesita revisar."
            )
            logger.info(msg)
            raise ValueError(msg)

        self.model = None  # se actualiza cuando se entrena el modelo

    def sequential_train_and_forecast(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
         Entrenamiento y predicción del modelo de consumo de gas a nivel nacional (diario)
         El modelo se entrena y predice en cada paso, avanzando en el tiempo de forma secuencial
         """

        fechas_validacion = self.df.index[-self.n_days_to_sequential_validation:]
        output_model = []

        for fecha_val in fechas_validacion:
            train_data = self.df[self.df.index < fecha_val]

            if train_data.empty:  # mínimo para descomponer
                continue

            self.__model_fitted(train_data)
            # Forecast t+1 como último valor de tendencia + estacionalidad
            trend_forecast = self.model.trend.iloc[-1]
            season_forecast = self.model.seasonal.iloc[-1]
            y_pred = trend_forecast + season_forecast
            logger.info("Realización de las predicciones del modelo")

            y_test = self.df.loc[fecha_val, self.target]
            mae, mape = self._get__metrics(y_test, y_pred)

            # obtención naive model
            y_naive, mae_naive, mape_naive = self._get_naive_predictions(fecha_val, y_test)

            output_model.append({
                'fecha_predicha': fecha_val,
                'y': y_test,
                'y_hat': y_pred,
                'y_hat_naive': y_naive,
                'mae': mae,
                'mape': mape,
                'mae_naive': mae_naive,
                'mape_naive': mape_naive
            })
            logger.info("Obtención de resultados: métricas y predicciones")
        logger.info("No disponible importancia de variables en la Descomposición Temporal")
        return pd.DataFrame(output_model), pd.DataFrame()

    def __model_fitted(self, X_train: pd.DataFrame) -> None:
        """
        Entrenamiento de un modelo de descomposición STL partir de la muestra de entrenamiento
        """
        # STL descomposición
        stl = STL(X_train[self.target], period=self.period)  # o el periodo que toque
        self.model = stl.fit()
        logger.info("Descomposición STL realizada")
