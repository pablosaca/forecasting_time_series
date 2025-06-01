# Documentación

Este proyecto trata sobre el consumo de gas a nivel nacional con periodicidad diaria.
Se busca predecir su consumo al día siguiente (horizonte temporal de 1 día)

A lo largo de este proyecto se lleven a cabo una serie de etapas que se exponen a continuación:
- Carga de datos: desde ficheros se convierten a dataframes de pandas (consumo, meteo, info sobre vacaciones)
- Limpieza de datos: estudio de valores nulos en las distintas variables y su imputación
- Análisis series: estudio estadístico y visual de las series de tiempo tanto de consumo como de variables meteorológicas
- Generación de features: creación de variables para uso en el futuro modelo (variables temporales, lags, medias móviles, festivos, etc. )
- Modelización: diseño de dos modelos (ligthgbm y descomposición temporal) para analizar la bondad de ajuste y comparar. 
  - Durante el entrenamiento se lleva a cabo un proceso de simulación de predicción en real (se reentrena y predice diariamente)
  - Comparación de cada modelo lgbm/descomposición temporal con un modelo dummy (lag 1 del consumo)
  - Se guarda el artefacto del modelo y las variables explicativas utilizadas (modelo lgbm)
- Predicción: se carga el modelo y las variables previamente guardadas y realiza la predicción para el día t+1 

Se proporciona una serie de módulos con sus respectivas clases, métodos y funciones:

```

│
├── README.md                                       # Lee el fichero                             
│                              
├── src  
│   └── model                                       # código fuente para el entrenamiento
│         └── model.py                             
│
│    └── predict                                    # código fuente para la predicción
│         └── predict.py
│
│     └──preprocessing                              # código fuente para la limpieza de las series y la generación de features
│         └── clean_data.py                           
│         └── feature_engineer.py
│
│   └── utils                                       # código fuente para la carga de datos, visualización y otros 
│         └── graphs.py                           
│         └── load_data.py
│         └── logger.py                           
│         └── utils.py
│
├── notebooks  
│   └── 01. Análisis y extracción características.ipynb
│   └── 02. Entrenamiento y bondad ajuste.ipynb    
│   └── 03. Modelo final.ipynb 
│   └── 04. Prediccion.ipynb    
│
├── data                                    # se encuentran disponibles ficheros csv como input del proyecto 
│ 
├── artefacto                               # se guarda el modelo (pickle) y las variables explicativas (json)
│ 
├── requirements.txt                        # requirements list

```

## Instalación

Creación de un entorno virtual del proyecto

```
conda create -n forecasting python=3.10
```

Para activar el entorno virtual usa la siguiente instrucción

```
conda activate forecasting
```

Así, instala las dependencias del fichero `requirements.txt` usando `pip`

```
pip install -r requirements.txt
```

