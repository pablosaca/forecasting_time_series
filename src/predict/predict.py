import joblib
import json
from typing import Optional, List
from src.utils.logger import get_logger

import numpy as np
import pandas as pd

logger = get_logger()


class PredictForecaster:

    def __init__(
            self,
            path: str,
            model_name: str,
            features_name: str,
            categorical_features: Optional[List[str]] = None,
    ):
        self.path = f"../{path}"
        self.model_name = model_name
        self.features_name = features_name
        aux_cat_feats = ["es_festivo", "dia_semana", "festivo_previo", "festivo_siguiente", "semana_anyo"]
        self.categorical_features = categorical_features if categorical_features is not None else aux_cat_feats

        self.model = None  # se carga el artefacto guardado en la ruta
        self.features = None  # se carga el artefacto guardado en la ruta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicción del modelo
        """
        self.__load_model()
        self.__load_features()
        X = X[self.features]
        X = self.__select_categorical_features(X)
        y_hat = self.model.predict(X)[0]
        logger.info("Predicción del modelo realizada")
        return y_hat

    def __select_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Se tienen en cuenta las variables categóricas
        """
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
                logger.info(f"La variable {col} es convertida a categórica")
        return df

    def __load_model(self) -> None:
        """
        Cargar el modelo
        """
        self.model = joblib.load(f"{self.path}/{self.model_name}.pkl")
        logger.info(f"Modelo cargadado desde {self.path}/{self.model_name}.pkl")

    def __load_features(self) -> None:
        """
        Cargar el modelo
        """
        with open(f"{self.path}/{self.features_name}.json", "r", encoding="utf-8") as file:
            self.features = json.load(file)["features"]
        logger.info(
            f"Cargadas las variables explicativas utilizadas en el modelo desde {self.path}/{self.features_name}.json"
        )
        logger.info(f"Variables del modelo: {self.features}")
