# README - Ejecución de Regresión en PySpark MLlib

## Descripción

Este script ejecuta un análisis de regresión en precios de acciones usando PySpark MLlib. El flujo incluye:

1. **Carga de datos**: Lee archivos CSV (con extensión `.txt`) desde `data/Stocks/`
2. **Preprocesado**: 
   - Añade características cíclicas (seno/coseno del día y mes)
   - Crea variable target `Close20` (precio de cierre 20 días después)
3. **Modelos entrenados**: Dos modelos diferentes con grid search
   - Linear Regression
   - Generalized Linear Regression
4. **Evaluación**: Métricas (RMSE, MAE, R²) guardadas en CSV en `resultados/`

---

## Requisitos Previos

### 1. Java instalado
```bash
java -version
```

Si no está instalado:
```bash
sudo apt-get update
sudo apt-get install default-jre default-jdk
```

### 2. Python 3.8+
```bash
python3 --version
```

### 3. PySpark y dependencias
Se recomienda crear un entorno virtual:

```bash
# Navegarse al directorio del proyecto
cd /home/user/path/to/iage_prac_5

# Crear entorno virtual
python3 -m venv venv

# Activar el entorno
source venv/bin/activate  # En Linux/Mac
# Windows: venv\Scripts\activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install pyspark==3.5.0 pandas numpy
```

### 4. Estructura de directorios correcta
Asegurate de que la estructura sea:
```
iage_prac_5/
├── main.py
├── data/
│   ├── Stocks/
│   │   ├── aapl.us.txt
│   │   ├── msft.us.txt
│   │   ├── ... (más archivos .txt)
├── resultados/  (se crea automáticamente)
└── README_ejecucion.md
```

---

## Preparación de Datos

Los archivos de datos deben:
- Estar ubicados en `data/Stocks/`
- Ser archivos CSV con extensión `.txt`
- Tener la estructura: `Date,Open,High,Low,Close,Volume,OpenInt`

Si los datos están en otra ubicación, asegúrate de copiarlos:
```bash
mkdir -p data/Stocks
cp /path/to/stock/files/*.txt data/Stocks/
```

---

## Ejecución del Script

### Opción 1: Ejecución por defecto (Local)

```bash
# Asegúrese de estar en la carpeta del proyecto
cd /path/to/iage_prac_5

# Activar el entorno virtual
source venv/bin/activate

# Ejecutar el script
python3 main.py
```

### Opción 2: Especificar número de particiones

Para aprovechar múltiples nodos en un cluster:

```bash
python3 main.py --num-partitions 12
```

### Opción 3: Ejecución en Cluster con Spark Standalone

```bash
# Requisito: Spark está instalado y configurado
python3 main.py \
    --master spark://master-node:7077 \
    --num-partitions 12 \
    --app-name StockRegression
```

### Opción 4: Ejecución con Yarn (si disponible)

```bash
python3 main.py \
    --master yarn \
    --num-partitions 16 \
    --num-workers 3
```

### Opción 5: Todos los parámetros personalizados

```bash
python3 main.py \
    --master local[8] \
    --num-partitions 8 \
    --app-name MyStockAnalysis \
    --output-dir custom_results
```

---

## Parámetros del Script

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--num-partitions` | int | 4 | Número de particiones para paralelización en Spark |
| `--app-name` | str | StockPriceRegression | Nombre de la aplicación Spark |
| `--master` | str | local[*] | URL del master Spark (local[*], spark://host:7077, yarn, etc.) |
| `--num-workers` | int | 1 | Número de workers/nodos (para modo cluster) |
| `--output-dir` | str | resultados | Directorio de salida para resultados CSV |

---

## Salida Esperada

### 1. Consola
El script imprimirá:
```
================================================================================
PySpark MLlib - Stock Price Regression with Grid Search
================================================================================

Configuration:
  - Master: local[*]
  - App Name: StockPriceRegression
  - Num Partitions: 4
  - Output Directory: resultados

STEP 1: Loading stock data...
Loaded data with X rows

STEP 2: Adding cyclic date features...
STEP 3: Adding target variable (Close20)...
...
================================================================================
BEST MODELS (by RMSE):
...
```

### 2. Archivos de Salida
En la carpeta `resultados/`:
- `regression_results_YYYYMMDD_HHMMSS.csv`

Con columnas:
- `model_name`: Nombre del modelo (LinearRegression, GeneralizedLinearRegression)
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `r2`: Coeficiente de determinación
- `seed`: Semilla aleatoria utilizada
- `regParam`: Parámetro de regularización
- `elasticNetParam` (solo Linear Regression)
- `maxIter` (solo Generalized Linear Regression)

---

## Troubleshooting

### Problema: "No such file or directory: 'data/Stocks'"
**Solución**: Verifica que estés en el directorio correcto y que los archivos de datos existan:
```bash
cd /path/to/iage_prac_5
ls data/Stocks/
```

### Problema: "ModuleNotFoundError: No module named 'pyspark'"
**Solución**: Asegúrate de activar el entorno virtual e instalar PySpark:
```bash
source venv/bin/activate
pip install pyspark
```

### Problema: Java no está instalado
**Solución**: Instala Java:
```bash
sudo apt-get install default-jre default-jdk
```

### Problema: El script es muy lento
**Solución**: 
- Aumenta el número de particiones: `--num-partitions 16`
- Comprueba que tienes suficiente RAM disponible
- En máquinas virtuales, asigna más recursos al contenedor

### Problema: OutOfMemory
**Solución**: Modifica la memoria en el script o en el comando de ejecución:
```bash
python3 main.py --num-partitions 4  # Comenzar con menos particiones
```

O edita las líneas en main.py:
```python
.config("spark.driver.memory", "4g")    # Aumenta si disponible
.config("spark.executor.memory", "4g")  # Aumenta si disponible
```

---

## Configuración en Vagrant

Si estás ejecutando en máquinas virtuales con Vagrant (3 workers + 1 master):

### 1. Copiar los datos al master/driver

```bash
# Desde la máquina host
vagrant scp --machine master . :/home/vagrant/iage_prac_5

# O manualmente dentro de la VM
scp -r /local/path/iage_prac_5 vagrant@master-ip:/home/vagrant/
```

### 2. Ejecutar en el master

```bash
# SSH al master
vagrant ssh master

# O directamente
ssh vagrant@master-ip

# Navegar al directorio
cd /home/vagrant/iage_prac_5

# Activar entorno y ejecutar
source venv/bin/activate
python3 main.py --num-partitions 12
```

### 3. Para ejecución distribuida con Spark Standalone

**En el master:**
```bash
# Iniciar Spark master
$SPARK_HOME/sbin/start-master.sh

# Verificar que está corriendo
jps

# En el master node, obtener la URL:
# Busca el puerto (típicamente 7077 o similar)
```

**En cada worker:**
```bash
# Iniciar Spark worker (reemplaza MASTER_URL)
$SPARK_HOME/sbin/start-worker.sh spark://master:7077
```

**Ejecutar el script en el master:**
```bash
python3 main.py \
    --master spark://master:7077 \
    --num-partitions 12 \
    --app-name StockAnalysisCluster
```

---

## Detalles del Algoritmo

### Preprocesado
1. **Características cíclicas**: 
   - `DateMonthSin = sin((mes - 1) / 12 * 2π)`
   - `DateMonthCos = cos((mes - 1) / 12 * 2π)`
   - `DateDaySin = sin((día - 1) / 31 * 2π)`
   - `DateDayCos = cos((día - 1) / 31 * 2π)`

2. **Target**:
   - `Close20 = Close[t+20]` (precio de cierre 20 días después)

3. **Normalización**: Features están estandarizadas (media=0, std=1)

### Grid Search

#### Linear Regression
- **regParam**: [0.0, 0.01, 0.1] (0=sin regularización, ↑=más regularización)
- **elasticNetParam**: [0.0, 0.5, 1.0] (0=Ridge, 1=Lasso)
- **Seeds**: [42, 123, 456] (para reproducibilidad con variación)
- **Total combinaciones**: 3 × 3 × 3 = 27

#### Generalized Linear Regression
- **regParam**: [0.0, 0.01, 0.1]
- **maxIter**: [50, 100]
- **Seeds**: [42, 123, 456]
- **Total combinaciones**: 3 × 2 × 3 = 18

---

## Métricas de Evaluación

- **RMSE** (Root Mean Squared Error): Penaliza más errores grandes. Útil para detectar outliersError RMS. Cuanto menor, mejor.
- **MAE** (Mean Absolute Error): Error promedio en valor absoluto. Robusto a outliers. Cuanto menor, mejor.
- **R²** (Coeficiente de Determinación): Proporción de varianza explicada [0, 1]. Cuanto mayor, mejor.

---

## Notas Importantes

- El script es reproducible cuando se especifica `--seed`
- Los datos se particionan para paralelización automática
- El train/test split es estratificado (20% test, 80% train)
- Los resultados se guardan automáticamente con timestamp
- Revisa el archivo CSV en `resultados/` para comparar modelos

---

## Contacto / Soporte

Si encuentras problemas durante la ejecución, verifica:
1. Que PySpark está instalado: `python3 -c "import pyspark; print(pyspark.__version__)"`
2. Que Java está disponible: `java -version`
3. Que los datos están en la ubicación correcta: `ls data/Stocks/ | head`
4. Los logs de PySpark en la salida de consola
