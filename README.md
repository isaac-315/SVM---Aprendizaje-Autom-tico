# Práctica: Forecasting con SVM (SVR) e integración con MLflow

## Descripción del proyecto

Este proyecto desarrolla un modelo de **Máquinas de Soporte Vectorial para Regresión (SVR)** aplicado al pronóstico de demanda de productos en un contexto de retail.

Se utiliza el dataset `retail_unificado.csv` para entrenar un modelo capaz de estimar la demanda del día siguiente a partir del historial de ventas de los siete días anteriores.

Además, se implementa un proceso de optimización de hiperparámetros (*Fine-Tuning*) y se utiliza **MLflow** para registrar, monitorear y comparar los experimentos realizados.

## Información académica

- **Materia:** Aprendizaje Automático
- **Grupo:** N.º 2 - Computación
- **Integrantes:** 
  - Leo Vásconez
  - Isaac Mora
  - Jean Pierre Valarezo
- **Profesor:** Ing. Remigio Hurtado
- **Fecha:** 16 de noviembre de 2026

## Objetivos

- Implementar un modelo de regresión utilizando Máquinas de Soporte Vectorial (SVR).
- Preparar los datos de ventas mediante ventanas temporales (*Lags*).
- Aplicar estandarización a las variables de entrada y salida.
- Entrenar un modelo SVR base.
- Optimizar los hiperparámetros mediante `GridSearchCV`.
- Registrar los experimentos, parámetros y métricas utilizando MLflow.
- Comparar el enfoque SVR con el modelo de Red Neuronal Recurrente (RNN) desarrollado anteriormente.

## Tecnologías utilizadas

| Tecnología | Descripción |
|---|---|
| Python | Lenguaje de programación |
| Pandas | Manipulación y análisis de datos |
| NumPy | Operaciones numéricas |
| Scikit-learn | Implementación y evaluación del modelo SVR |
| MLflow | Seguimiento y registro de experimentos |
| Jupyter Notebook | Desarrollo y ejecución del proyecto |

## Estructura del repositorio

```text
SVM/
│
├── practica_svm_AprendizajeAutomatico.ipynb
├── retail_unificado.csv
└── README.md
```

## Desarrollo del proyecto

### 1. Preparación de los datos

Se carga el dataset `retail_unificado.csv` y se agrupan las ventas por fecha y producto.

Se aplica una técnica de ingeniería de características (*Feature Engineering*) mediante ventanas temporales:

- `Lag_1`: ventas del día anterior.
- `Lag_2`: ventas de hace dos días.
- `Lag_3`: ventas de hace tres días.
- `Lag_4`: ventas de hace cuatro días.
- `Lag_5`: ventas de hace cinco días.
- `Lag_6`: ventas de hace seis días.
- `Lag_7`: ventas de hace siete días.

Estas variables permiten transformar la serie temporal en un conjunto de datos tabular que puede ser utilizado por SVR.

### 2. Escalado y división de datos

Se utiliza `StandardScaler` para estandarizar las variables de entrada y la variable objetivo.

Posteriormente, se divide el conjunto de datos en entrenamiento y prueba, utilizando una proporción de 80 % y 20 %, respectivamente.

La división se realiza con `shuffle=False` para conservar el orden cronológico de las observaciones.

### 3. Entrenamiento del modelo base

Se implementa un modelo SVR utilizando el kernel `rbf` y los hiperparámetros predeterminados.

Este modelo funciona como referencia (*baseline*) para evaluar posteriormente los resultados de la optimización.

### 4. Optimización de hiperparámetros

Se utiliza `GridSearchCV` para encontrar la mejor combinación de hiperparámetros:

- **C:** controla la penalización de los errores.
- **Gamma:** determina el alcance de influencia de los puntos de entrenamiento.
- **Kernel:** define la función utilizada para transformar los datos; se evalúan `rbf` y `linear`.

La búsqueda utiliza validación cruzada de tres pliegues (`cv=3`) y el criterio `neg_mean_squared_error`.

### 5. Seguimiento de experimentos con MLflow

Se configura el experimento `Forecasting_SVR_Retail`.

Se registran dos ejecuciones:

- `SVR_Base`: modelo inicial.
- `SVR_Finetuning`: modelo optimizado.

Para cada ejecución se almacenan los hiperparámetros, las métricas de evaluación y el modelo entrenado.

Las métricas utilizadas son:

- **RMSE (Root Mean Squared Error):** mide el error cuadrático medio de las predicciones, dando mayor peso a errores grandes.
- **MAE (Mean Absolute Error):** mide el promedio de los errores absolutos entre las predicciones y los valores reales.

### 6. Predicción de nuevos datos

Se desarrolla la función `predecir_demanda_nuevo_sample()`, que utiliza el modelo SVR optimizado para estimar la demanda del día siguiente.

La función recibe:

- El identificador del producto.
- El historial de ventas de los últimos siete días.

Posteriormente, escala los datos, realiza la predicción, recupera la escala original y evita resultados negativos.

Se realizan pruebas con dos escenarios:

- Producto de alta rotación.
- Producto de baja rotación.

## Instalación y ejecución

### Requisitos previos

- Python 3.10 o superior.
- Jupyter Notebook o JupyterLab.

### 1. Clonar el repositorio

```bash
git clone https://github.com/isaac-315/SVM---Aprendizaje-Autom-tico.git
```

### 2. Ingresar al directorio

```bash
cd SVM---Aprendizaje-Autom-tico
```

### 3. Instalar las dependencias

```bash
pip install mlflow scikit-learn pandas numpy
```

### 4. Ejecutar el notebook

Abrir Jupyter Notebook:

```bash
jupyter notebook
```

Seleccionar el archivo:

`practica_svm_AprendizajeAutomatico.ipynb`

Ejecutar las celdas en orden, verificando que el archivo `retail_unificado.csv` se encuentre en el mismo directorio que el notebook.

## Comparación entre SVR y RNN

El proyecto también contempla una comparación conceptual con la Red Neuronal Recurrente (RNN) desarrollada en una práctica anterior.

- **SVR:** requiere transformar los datos temporales en características tabulares mediante ventanas de tiempo. Su optimización se realiza mediante la búsqueda de hiperparámetros.
- **RNN:** procesa secuencias temporales directamente y puede aprender patrones complejos mediante sus estados internos.

El SVR representa una alternativa para conjuntos de datos pequeños o medianos, mientras que las RNN pueden resultar apropiadas para problemas temporales de mayor complejidad.

## Referencias

- Raschka, S. (2017). *Python Machine Learning* (2nd ed.). Packt Publishing.
- MLflow. (2024). [Tracking Experiments](https://mlflow.org/docs/latest/tracking.html).

---

**Universidad Politécnica Salesiana**
