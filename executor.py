from src.utils.utils import get_model_df, save_model
from src.utils.logger import get_logger
from src.utils.load_data import BasicData
from src.preprocessing.clean_data import CleanData
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils.graphs import (
    plot_correlation_matrix, plot_acf_pacf, plot_multiple_time_series, seasonal_decompose_plot
)
from src.model.model import LGBMGasForecaster, STLForecaster
from src.predict.predict import PredictForecaster


logger = get_logger()


output_df = BasicData(
    consumption_tab_name="Consumption",
    meteo_tab_name="Meteorological_data_anon",
    holiday_city_tab="holyday_per_city_anon",
    inhabitants_tab="inhabiants_anon",
).get_basic_data()


clean_data = CleanData(output_df)
output_end_df = clean_data.basic_df  # tablón que ya tiene todas las fechas del histórico de consumo de gas
output_end_df = clean_data.rolling_average_impute_values(output_end_df, "Consumption", 3)
output_end_df = clean_data.rolling_average_impute_values(output_end_df, "ta")
output_end_df = clean_data.rolling_average_impute_values(output_end_df, "v")
output_end_df = clean_data.rolling_average_impute_values(output_end_df, "prec")

output_end_df = clean_data.formatting_columns(
    output_end_df,
    {
        "date_local_int": "fecha",
        "Consumption": "consumo_gas",
        "ta": "temperatura",
        "prec": "precipitacion",
        "v": "velocidad_viento",
        "nm_habs": "numero_habitantes"
    }
)

# Análisis de las series (consumo de gas, temperatura y velocidad del viento
seasonal_decompose_plot(output_end_df, "consumo_gas", 8)
seasonal_decompose_plot(output_end_df, "temperatura", 7)
seasonal_decompose_plot(output_end_df, "velocidad_viento", 7)
seasonal_decompose_plot(output_end_df, "precipitacion", 7)

plot_acf_pacf(output_end_df["consumo_gas"], lags=30, zero=False)
plot_acf_pacf(output_end_df["temperatura"], lags=30, zero=False)
plot_acf_pacf(output_end_df["velocidad_viento"], lags=30, zero=False)

plot_multiple_time_series(output_end_df, "fecha", ["consumo_gas", "temperatura"])
plot_multiple_time_series(output_end_df, "fecha", ["consumo_gas", "velocidad_viento"])
plot_multiple_time_series(output_end_df, "fecha", ["temperatura", "velocidad_viento"])
plot_multiple_time_series(output_end_df, "fecha", ["temperatura", "precipitacion"])

# lags:
# consumo:  1, 8, 15
# temperatura: 1, 14
# precipitacion: 1, 2
feature_enginieering = FeatureEngineer(
    target="consumo_gas",
    feats=["temperatura", "precipitacion", "velocidad_viento"],
    lags={
        "consumo_gas": [1, 2, 8, 15],
        "temperatura": [1, 14, 28],
        "precipitacion": [1, 2],
        "velocidad_viento": [1, 2]
    },
    windows=[3, 8, 15],
    shift=1
)
output_ft_df = feature_enginieering.get_date_features(output_end_df)
output_ft_df = feature_enginieering.get_lags_features(output_ft_df)
output_ft_df = feature_enginieering.get_rolling_windows_features(output_ft_df)

df = output_ft_df.drop(columns="fecha")
plot_correlation_matrix(df)

model_df = get_model_df(
    output_ft_df,
    save_columns=[
        "fecha",
        "consumo_gas", "consumo_gas_lag_1", "consumo_gas_lag_15", "consumo_gas_rolling_window_8",
        "consumo_gas_rolling_window_8_std", "consumo_gas_rolling_window_15_std",
        "temperatura_rolling_window_3", "temperatura_lag_14", "temperatura_lag_28",
        "velocidad_viento_rolling_window_3", "velocidad_viento_rolling_window_8", "velocidad_viento_rolling_window_15",
        "es_festivo", "dia_semana", "festivo_previo", "festivo_siguiente", "semana_anyo",
    ],
    path="data",
    name="model_data"
)

forecaster = LGBMGasForecaster(model_df, 30)
results_full_df, feature_importance_full_df = forecaster.sequential_train_and_forecast()

save_model(forecaster.model, "artefacto", "consumo_gas_model")

# proxy ejemplo predicción  obtención del dataframe para realizar predicción
X_df = model_df[model_df.index == "2021-05-03"].drop(columns="consumo_gas")
predict_model = PredictForecaster("artefacto", "consumo_gas_model")
y_pred = predict_model.predict(X_df)

logger.info("Simulación predicción finalizada")
