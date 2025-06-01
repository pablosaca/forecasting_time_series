import pandas as pd

from src.utils.logger import get_logger


logger = get_logger()


class BasicData:

    def __init__(
            self, consumption_tab_name: str, meteo_tab_name: str, holiday_city_tab: str, inhabitants_tab: str
    ):
        self.consumption = consumption_tab_name
        self.meteo = meteo_tab_name
        self.holiday = holiday_city_tab
        self.inhabitants = inhabitants_tab

    def get_basic_data(self) -> pd.DataFrame:
        """
        Obtención de tablón con target del problema y features asociadas

        En este método se cargan los ficheros y se genera el tablón base para continuar
        """
        consumption_df = self.__load_data(self.consumption)
        meteo_df = self.__load_data(self.meteo)
        holidays_df = self.__load_data(self.holiday)
        inhabiants_anon_df = self.__load_data(self.inhabitants)
        meteo_agg_df = self.__preprocessing_meteorologycal_data(meteo_df)
        output_df = self.__merge_tables(consumption_df, meteo_agg_df, ["date_local_int"])
        holidays_inhabiants_anon_df = self.__merge_tables(inhabiants_anon_df, holidays_df, ["cod_an"])
        habitants_holidays_df = self.__preprocessing_habitants_holyday_data(holidays_inhabiants_anon_df)
        output_df = self.__merge_tables(output_df, habitants_holidays_df, ["date_local_int"])
        logger.info(
            "Obtención del macrotablón base. Incluye consumo gas + features (variables climatológica como festivos"
        )
        return output_df

    @staticmethod
    def __load_data(table_name: str) -> pd.DataFrame:
        """
        Carga fichero csv como pandas DataFrame
        """
        logger.info(f"Se carga el fichero {table_name}")
        return pd.read_csv(f"../data/{table_name}.csv")

    @staticmethod
    def __preprocessing_meteorologycal_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtención de variables climáticas a nivel diario

        El valor diario se calcula como el promedio de la información disponible de forma horaria.
        En el caso de la variable temperatura se obtiene el promedio de temperatura máxima y mínima
        """
        df = df.assign(ta=(df["tamax"] + df["tamin"]) / 2)
        df = df[["date_local_int", "ta", "prec", "vmax", "inso"]]
        agg_df = df.groupby("date_local_int").agg(
            {"ta": "mean", "prec": "mean", "vmax": "mean"}
        ).reset_index()
        agg_df = agg_df.rename(columns={"vmax": "v"})
        logger.info("Se obtiene información meteorológica (temperatura, preciptiación, viento a nivel diario)")
        return agg_df

    @staticmethod
    def __preprocessing_habitants_holyday_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtención del número de habitantes que hay en día de festivo (local, provincial, comunitario o nacional)
        """
        logger.info("Obtención del número de habitantes que están de vacaciones en los días de festivo")
        return df.groupby("date_local_int")["nm_habs"].count().reset_index()

    @staticmethod
    def __merge_tables(df1: pd.DataFrame, df2: pd.DataFrame, cols_merge: list) -> pd.DataFrame:
        """
        Cruce de tablas
        """
        logger.info(f"Se cruzan dos tablas (left-join) por {cols_merge}")
        return pd.merge(df1, df2, on=cols_merge, how="left")
