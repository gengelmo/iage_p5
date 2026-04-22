Este es un análisis técnico profundo del script de PySpark proporcionado. El código implementa un pipeline de aprendizaje automático distribuido para la predicción de retornos bursátiles, utilizando **Spark MLlib** y optimizando la ejecución mediante el aprovechamiento del paralelismo de Spark.

---

## 1. Configuración del Entorno y Detección del Master
El script comienza con una lógica robusta para determinar dónde y cómo se ejecutará el clúster de Spark.

### Detección Dinámica (`detect_master_argument`)
Antes de iniciar la `SparkSession`, el código busca en variables de entorno (`SPARK_MASTER_URL`, `SPARK_MASTER_HOST`) y en archivos de configuración locales (`spark-defaults.conf`).
* **Estrategia de paralelización:** Al permitir la detección de `yarn` o `spark://`, el script se adapta para distribuir la carga en un clúster multi-nodo. Si no encuentra configuración, recurre a `socket.gethostname()`, asumiendo que se ejecuta en el nodo maestro de un clúster standalone.

### Creación de la sesión (`create_spark_session`)
Se utiliza el patrón **Builder**. Dos configuraciones son críticas aquí:
* `spark.sql.shuffle.partitions`: Establecido en 200 (valor por defecto de Spark). Esto determina cuántas particiones se crean durante operaciones de "shuffle" (como los joins o las funciones de ventana en las series temporales).
* **Gestión de Memoria:** Define explícitamente la memoria del *driver* y del *executor*, lo cual es vital para evitar errores de `OutOfMemory` (OOM) cuando se recolectan resultados de la validación cruzada.

---

## 2. Ingestión de Datos Masiva y Paralelizada (`load_stock_data`)
Este paso es fundamental para el rendimiento. En lugar de iterar sobre archivos (lo cual sería un antipatrón en Spark), se utiliza una **lectura masiva**.

### Uso de Esquema Explícito (`StructType`)
Spark, por defecto, puede intentar inferir el esquema leyendo los datos dos veces. Al proporcionar un `schema` definido, se elimina esa pasada adicional, reduciendo el I/O de disco a la mitad.

### Lectura Distribuida de Archivos
* `spark.read.csv("data/Stocks/*.txt")`: Spark utiliza un listado de archivos global y asigna diferentes archivos a diferentes *executors*. Si hay 1000 archivos, los trabajadores los leerán en paralelo.
* `F.input_file_name()`: Esta es una función nativa de Spark SQL. Permite obtener el origen de cada fila sin necesidad de procesar los archivos uno a uno en el driver. Se ejecuta de forma distribuida en los nodos que están leyendo los datos.
* `F.regexp_extract`: Se aplica una expresión regular sobre la columna de la ruta para extraer el símbolo de la acción. Al ser una función de Spark SQL, se ejecuta en los *executors* como una transformación de columna.

---

## 3. Ingeniería de Características (Feature Engineering)
El script aplica transformaciones matemáticas y temporales utilizando el motor de **Spark SQL Catalyst Optimizer**.

### Características Cíclicas (`add_cyclic_features`)
Se transforman el mes y el día en componentes seno y coseno. Esto permite que el modelo entienda la continuidad temporal (que diciembre está cerca de enero). Se utilizan funciones de `pyspark.sql.functions` que se traducen directamente a expresiones de columna de bajo nivel.

### Funciones de Ventana (`add_target_variable` y `add_past_returns`)
Aquí se utilizan las **Window Functions** de Spark, esenciales para series temporales:
* `Window.partitionBy("Symbol").orderBy("Date")`: Esto define cómo se agrupan los datos para el cálculo. Spark garantiza que todos los datos de un mismo `Symbol` terminen en la misma partición para poder calcular los retardos (`lag`) y adelantos (`lead`).
* `F.lead("Close", 20)`: Busca el precio 20 días en el futuro para definir el *target*.
* `F.lag("Close", n)`: Busca precios pasados para calcular retornos históricos.
* **Nota sobre Rendimiento:** Las funciones de ventana pueden causar un *shuffle* masivo de datos a través de la red si hay muchos símbolos. El script maneja esto implícitamente mediante el particionamiento de Spark.

---

## 4. Preparación para MLlib (`prepare_features_and_target`)
Spark MLlib requiere un formato específico de datos.

* **Limpieza de Nulos:** Se eliminan filas donde las ventanas de tiempo no pudieron completarse (por ejemplo, los primeros 10 días de una acción no tienen `PastReturn10`).
* **Renombrado a `label`:** MLlib busca por defecto una columna llamada `label` para el objetivo de la regresión.

---

## 5. División de Datos y Caché (`create_train_test_split`)
* `randomSplit([0.8, 0.2])`: Realiza una división estocástica distribuida. A diferencia de Scikit-Learn, esto ocurre en los nodos del clúster.
* **`train_df.cache()` / `test_df.cache()`**: Este es un paso de optimización crítico. Al llamar a `cache()`, los datos se almacenan en la memoria de los *executors* (en formato RDD o columnar). Como el proceso de entrenamiento con **Grid Search** leerá estos datos decenas de veces, tenerlos en memoria evita volver a leer los archivos de disco y re-procesar todas las funciones de ventana en cada iteración del modelo.

---

## 6. Modelado y Entrenamiento con Spark MLlib
El script utiliza dos modelos: `LinearRegression` (LR) y `GeneralizedLinearRegression` (GLR). Aquí es donde reside la mayor parte de la lógica de MLlib.

### El Pipeline de MLlib
Se utiliza `pyspark.ml.Pipeline`, que es un flujo de trabajo que organiza `Estimators` y `Transformers`. Los componentes son:
1.  **`VectorAssembler`**: Spark MLlib no acepta múltiples columnas de entrada. Este transformador concatena las columnas de características en un único vector denso (`features`).
2.  **`StandardScaler`**: Normaliza las características. Es crucial que esté dentro del pipeline. Esto asegura que los parámetros de escalado (media y desviación) se calculen solo sobre los datos de entrenamiento de cada *fold* de la validación cruzada, evitando el **data leakage** (filtración de información del test/validación al entrenamiento).
3.  **Modelos (`LR` / `GLR`)**: Los algoritmos de regresión propiamente dichos.

### Grid Search y Validación Cruzada Paralelizada
El uso de `CrossValidator` y `ParamGridBuilder` es la clave de la automatización:
* `ParamGridBuilder`: Define el hiperespacio de búsqueda (parámetros de regularización, iteraciones, etc.).
* **`CrossValidator`**: Realiza una validación cruzada de $k$ pliegues ($numFolds=3$).
* **Estrategia de Paralelización (`parallelism=4`)**: Esta es una característica avanzada de Spark MLlib. El `CrossValidator` puede entrenar múltiples combinaciones de parámetros simultáneamente. Si el clúster tiene suficientes recursos (vCores), Spark lanzará múltiples tareas de entrenamiento de modelos en paralelo. Un valor de 4 significa que se entrenarán 4 configuraciones de hiperparámetros al mismo tiempo, acelerando drásticamente el proceso de búsqueda en comparación con una ejecución secuencial.

### Evaluación (`RegressionEvaluator`)
Se utiliza un evaluador nativo de Spark que calcula métricas como RMSE, MAE y $R^2$. Al ser un evaluador de MLlib, el cálculo de estas métricas sobre el conjunto de test también se realiza de forma distribuida.

---

## 7. Persistencia y Resumen de Resultados
Finalmente, el script recolecta los resultados de los modelos y los parámetros probados.
* `cv_model.avgMetrics`: Contiene el promedio del error (RMSE en este caso) de cada combinación del grid tras la validación cruzada.
* **Interoperabilidad con Pandas:** Al final, el script utiliza `pd.DataFrame(all_results)` y `to_csv`. Aquí se produce una pequeña "recolección" de datos hacia el driver. Dado que solo se recolectan las métricas finales y los nombres de los parámetros (no los datos de entrenamiento), esto es seguro y eficiente para generar informes locales.

---

## Resumen de Estrategias de Paralelización Utilizadas
1.  **Lectura de Datos:** Paralelismo a nivel de sistema de archivos (Data Source V2).
2.  **Transformaciones:** Paralelismo de datos mediante el motor SQL/DataFrames (Lazy evaluation y Catalyst).
3.  **Window Functions:** Re-particionamiento de datos por clave (`Symbol`) para procesamiento local en nodos.
4.  **Entrenamiento:** Paralelismo de modelos a través del parámetro `parallelism` en `CrossValidator`, permitiendo que el clúster entrene múltiples redes o regresiones de forma concurrente.
5.  **Caché:** Persistencia en memoria distribuida para evitar re-computación en algoritmos iterativos.
