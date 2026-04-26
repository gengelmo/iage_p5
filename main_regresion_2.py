"""
PySpark MLlib Script for Stock Return Prediction with Temporal Validation.
Trains regression models (Linear Regression, Generalized Linear Regression,
and Random Forest Regressor) on stock market data using temporal
train/validation/test splits.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import math
import re
import socket

import pandas as pd
from pyspark.sql import SparkSession, Window, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator




def validate_master_argument(master: str):
    """Validate --master to provide user-friendly errors before Spark startup."""
    if not master or not master.strip():
        raise ValueError("--master cannot be empty. Use local[*], spark://host:7077, or yarn.")

    master = master.strip()

    # Common placeholder values copied from templates/docs.
    if "MASTER_HOST" in master or "<" in master or ">" in master:
        raise ValueError(
            "Invalid --master value: placeholder detected. "
            "Replace it with the real host, e.g. spark://master:7077"
        )

    if master.startswith("spark://"):
        # Accept forms like spark://host:7077 or spark://host1:7077,host2:7077.
        if not re.fullmatch(r"spark://[^\s:]+:\d+(,[^\s:]+:\d+)*", master):
            raise ValueError(
                "Invalid standalone Spark URL format. Expected spark://host:7077 "
                "(or spark://host1:7077,host2:7077)."
            )

    return master


def _read_spark_master_from_defaults():
    """Read spark.master from spark-defaults.conf if available."""
    candidate_dirs = []

    spark_conf_dir = os.getenv("SPARK_CONF_DIR")
    spark_home = os.getenv("SPARK_HOME")

    if spark_conf_dir:
        candidate_dirs.append(Path(spark_conf_dir))
    if spark_home:
        candidate_dirs.append(Path(spark_home) / "conf")

    candidate_dirs.extend([
        Path("/opt/spark/conf"),
        Path("/usr/local/spark/conf"),
    ])

    for conf_dir in candidate_dirs:
        conf_file = conf_dir / "spark-defaults.conf"
        if not conf_file.exists():
            continue

        try:
            with conf_file.open("r", encoding="utf-8") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == "spark.master":
                        return parts[1]
        except OSError:
            continue

    return None


def detect_master_argument():
    """Auto-detect Spark master URL for executions on the master host."""
    master_url = os.getenv("SPARK_MASTER_URL")
    if master_url:
        return validate_master_argument(master_url)

    master = os.getenv("SPARK_MASTER")
    if master:
        return validate_master_argument(master)

    master_host = os.getenv("SPARK_MASTER_HOST")
    if master_host:
        master_port = os.getenv("SPARK_MASTER_PORT", "7077")
        return validate_master_argument(f"spark://{master_host}:{master_port}")

    master_from_defaults = _read_spark_master_from_defaults()
    if master_from_defaults:
        return validate_master_argument(master_from_defaults)

    # Last resort when running directly in the cluster master machine.
    host = socket.gethostname()
    return validate_master_argument(f"spark://{host}:7077")


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train regression models on stock market return data using PySpark MLlib"
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=4,
        help="Number of partitions for Spark (default: 4)",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="StockReturnRegression",
        help="Spark application name (default: StockReturnRegression)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads/nodes (for Yarn/Cluster mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resultados",
        help="Output directory for results CSV (default: resultados)",
    )
    
    return parser.parse_args()


def create_spark_session(app_name, master):
    """Create and return a Spark session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        .getOrCreate()
    )
    return spark

def load_stock_data(data_dir: str, num_partitions: int):
    """Carga eficiente de archivos CSV usando lectura masiva y funciones nativas."""
    spark = SparkSession.getActiveSession()
    
    # Definir el esquema explícitamente evita que Spark tenga que leer los archivos dos veces
    # Ajusta los nombres/tipos si varían, pero esto es lo estándar para el dataset de Kaggle
    schema = T.StructType([
        T.StructField("Date", T.StringType(), True),
        T.StructField("Open", T.DoubleType(), True),
        T.StructField("High", T.DoubleType(), True),
        T.StructField("Low", T.DoubleType(), True),
        T.StructField("Close", T.DoubleType(), True),
        T.StructField("Volume", T.DoubleType(), True),
        T.StructField("OpenInt", T.DoubleType(), True)
    ])

    print(f"Cargando datos masivamente desde {data_dir}...")

    # 1. Leer todos los archivos .txt del directorio de una sola vez
    # Usamos recursiveFileLookup si hay subcarpetas
    df = (spark.read
          .option("header", "true")
          .schema(schema)
          .csv(f"{data_dir}/*.txt"))

    # 2. Extraer el 'Symbol' del nombre del archivo de forma distribuida
    # F.input_file_name() devuelve la ruta completa, extraemos el nombre del archivo
    df = df.withColumn("FilePath", F.input_file_name())
    # Regex para extraer el nombre del archivo sin la ruta ni la extensión .us.txt
    df = df.withColumn("Symbol", F.regexp_extract("FilePath", r"([^/]+)\.us\.txt$", 1))
    df = df.drop("FilePath")

    # 3. Limpieza: Filtrar filas donde todas las columnas numéricas sean nulas 
    # (Equivalente a tu validación de archivos vacíos o corruptos)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Reparticionar y procesar tipos finales
    df = df.repartition(num_partitions)
    df = df.withColumn("Date", F.to_timestamp("Date"))
    
    # Forzar una acción para disparar la carga (opcional para debug)
    # total_rows = df.count()
    # print(f"Dataset total cargado con {total_rows} filas.")
    
    return df

def clean_market_data(df):
    """Apply minimal cleaning for market data consistency."""
    
    # Keep only valid positive OHLC prices and non-negative volume
    df = df.filter(
        (F.col("Open") > 0) &
        (F.col("High") > 0) &
        (F.col("Low") > 0) &
        (F.col("Close") > 0) &
        (F.col("Volume") >= 0)
    )

    # Basic OHLC consistency checks
    df = df.filter(
        (F.col("High") >= F.col("Open")) &
        (F.col("High") >= F.col("Close")) &
        (F.col("High") >= F.col("Low")) &
        (F.col("Low") <= F.col("Open")) &
        (F.col("Low") <= F.col("Close"))
    )

    return df

def filter_symbols_by_min_rows(df, min_rows=60):
    """Keep only symbols with at least min_rows observations."""
    valid_symbols = (
        df.groupBy("Symbol")
        .count()
        .filter(F.col("count") >= min_rows)
        .select("Symbol")
    )
    return df.join(valid_symbols, on="Symbol", how="inner")


def add_cyclic_features(df):
    """Add cyclic features for date components."""
    
    # Month
    month_normalized = (F.month("Date") - 1) / 12.0 * 2 * math.pi
    df = df.withColumn("DateMonthSin", F.sin(month_normalized))
    df = df.withColumn("DateMonthCos", F.cos(month_normalized))
    
    # Day of month
    day_normalized = (F.dayofmonth("Date") - 1) / 31.0 * 2 * math.pi
    df = df.withColumn("DateDaySin", F.sin(day_normalized))
    df = df.withColumn("DateDayCos", F.cos(day_normalized))

    # Day of week
    dow_normalized = (F.dayofweek("Date") - 1) / 7.0 * 2 * math.pi
    df = df.withColumn("DayOfWeekSin", F.sin(dow_normalized))
    df = df.withColumn("DayOfWeekCos", F.cos(dow_normalized))
    
    return df


def add_target_variable(df):
    """Add LogReturn20 target: log-return of closing price 20 days in the future."""
    
    window_spec = Window.partitionBy("Symbol").orderBy("Date")
    
    # Create lead to get closing price 20 days ahead and calculate log-return
    df = df.withColumn(
        "LogReturn20",
        F.log(F.lead("Close", 20).over(window_spec) / F.col("Close"))
    )
    
    return df


def add_market_features(df):
    """Add market-based predictive features using rolling windows by symbol."""
    
    window_spec = Window.partitionBy("Symbol").orderBy("Date")
    rolling_5 = window_spec.rowsBetween(-4, 0)
    rolling_10 = window_spec.rowsBetween(-9, 0)
    rolling_20 = window_spec.rowsBetween(-19, 0)
    rolling_60 = window_spec.rowsBetween(-59, 0)

    # Lags útiles
    close_lag_1 = F.lag("Close", 1).over(window_spec)
    close_lag_2 = F.lag("Close", 2).over(window_spec)
    close_lag_5 = F.lag("Close", 5).over(window_spec)
    close_lag_10 = F.lag("Close", 10).over(window_spec)
    close_lag_20 = F.lag("Close", 20).over(window_spec)
    close_lag_40 = F.lag("Close", 40).over(window_spec)
    volume_lag_1 = F.lag("Volume", 1).over(window_spec)

    # Retornos simples
    df = df.withColumn("ret_1", (F.col("Close") - close_lag_1) / close_lag_1)
    df = df.withColumn("ret_2", (F.col("Close") - close_lag_2) / close_lag_2)
    df = df.withColumn("ret_5", (F.col("Close") - close_lag_5) / close_lag_5)
    df = df.withColumn("ret_10", (F.col("Close") - close_lag_10) / close_lag_10)
    df = df.withColumn("ret_20", (F.col("Close") - close_lag_20) / close_lag_20)
    df = df.withColumn("ret_40", (F.col("Close") - close_lag_40) / close_lag_40)

    # Volatilidad histórica
    df = df.withColumn("vol_5", F.stddev("ret_1").over(rolling_5))
    df = df.withColumn("vol_10", F.stddev("ret_1").over(rolling_10))
    df = df.withColumn("vol_20", F.stddev("ret_1").over(rolling_20))

    # Ratio de volatilidad corto/largo
    df = df.withColumn(
        "vol_ratio_5_20",
        F.when(F.col("vol_20") != 0, F.col("vol_5") / F.col("vol_20"))
    )

    # Medias móviles
    df = df.withColumn("ma_5", F.avg("Close").over(rolling_5))
    df = df.withColumn("ma_10", F.avg("Close").over(rolling_10))
    df = df.withColumn("ma_20", F.avg("Close").over(rolling_20))
    df = df.withColumn("ma_60", F.avg("Close").over(rolling_60))

    df = df.withColumn("ma_ratio_5", F.col("Close") / F.col("ma_5"))
    df = df.withColumn("ma_ratio_10", F.col("Close") / F.col("ma_10"))
    df = df.withColumn("ma_ratio_20", F.col("Close") / F.col("ma_20"))
    df = df.withColumn("ma_ratio_60", F.col("Close") / F.col("ma_60"))

    # Intradía
    df = df.withColumn(
        "intraday_range",
        (F.col("High") - F.col("Low")) / F.col("Close")
    )

    df = df.withColumn(
        "open_close_change",
        (F.col("Close") - F.col("Open")) / F.col("Open")
    )

    df = df.withColumn(
        "high_close_ratio",
        (F.col("High") - F.col("Close")) / F.col("Close")
    )

    df = df.withColumn(
        "close_low_ratio",
        (F.col("Close") - F.col("Low")) / F.col("Close")
    )

    # Gap de apertura frente al cierre previo
    df = df.withColumn(
        "gap_open",
        (F.col("Open") - close_lag_1) / close_lag_1
    )

    # Volumen
    df = df.withColumn("volume_ma_10", F.avg("Volume").over(rolling_10))
    df = df.withColumn("volume_ma_20", F.avg("Volume").over(rolling_20))

    df = df.withColumn(
        "volume_ratio_10",
        F.when(F.col("volume_ma_10") != 0, F.col("Volume") / F.col("volume_ma_10"))
    )

    df = df.withColumn(
        "volume_ratio_20",
        F.when(F.col("volume_ma_20") != 0, F.col("Volume") / F.col("volume_ma_20"))
    )

    df = df.withColumn(
        "volume_change_1",
        F.when(volume_lag_1 != 0, (F.col("Volume") - volume_lag_1) / volume_lag_1)
    )

    df = df.withColumn(
        "log_volume",
        F.log1p(F.col("Volume"))
    )

    # Drawdown y posición dentro del rango reciente
    df = df.withColumn("rolling_max_10", F.max("Close").over(rolling_10))
    df = df.withColumn("rolling_max_20", F.max("Close").over(rolling_20))
    df = df.withColumn("rolling_min_20", F.min("Close").over(rolling_20))

    df = df.withColumn(
        "drawdown_10",
        (F.col("Close") - F.col("rolling_max_10")) / F.col("rolling_max_10")
    )

    df = df.withColumn(
        "drawdown_20",
        (F.col("Close") - F.col("rolling_max_20")) / F.col("rolling_max_20")
    )

    df = df.withColumn(
        "close_to_max_20",
        F.when(F.col("rolling_max_20") != 0, F.col("Close") / F.col("rolling_max_20"))
    )

    df = df.withColumn(
        "close_to_min_20",
        F.when(F.col("rolling_min_20") != 0, F.col("Close") / F.col("rolling_min_20"))
    )

    return df


def prepare_features_and_target(df):
    """Select and clean data for modeling (remove nulls)."""
    
    feature_cols = [
        # Temporales
        "DateMonthSin", "DateMonthCos",
        "DateDaySin", "DateDayCos",
        "DayOfWeekSin", "DayOfWeekCos",

        # Momentum / retornos
        "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "ret_40",

        # Volatilidad
        "vol_5", "vol_10", "vol_20", "vol_ratio_5_20",

        # Tendencia / medias móviles
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",

        # Intradía
        "intraday_range", "open_close_change",
        "high_close_ratio", "close_low_ratio", "gap_open",

        # Volumen
        "volume_ratio_10", "volume_ratio_20",
        "volume_change_1", "log_volume",

        # Drawdown / posición reciente
        "drawdown_10", "drawdown_20",
        "close_to_max_20", "close_to_min_20"
    ]
    
    df = df.dropna(subset=["LogReturn20"])
    df = df.dropna(subset=feature_cols)
    df = df.select(["Symbol", "Date"] + feature_cols + ["LogReturn20"])
    df = df.withColumnRenamed("LogReturn20", "label")
    
    return df, feature_cols


def create_train_validation_test_split_by_date(
    df,
    train_end="2015-01-01",
    val_end="2016-01-01"
):
    """Create temporal train/validation/test splits using date cutoffs."""
    
    train_end_ts = F.to_timestamp(F.lit(train_end))
    val_end_ts = F.to_timestamp(F.lit(val_end))

    train_df = df.filter(F.col("Date") < train_end_ts)
    val_df = df.filter((F.col("Date") >= train_end_ts) & (F.col("Date") < val_end_ts))
    test_df = df.filter(F.col("Date") >= val_end_ts)

    return train_df, val_df, test_df


def train_linear_regression_with_validation(train_df, val_df, test_df, feature_cols):
    """
    Train Linear Regression with manual grid search using validation set.
    """
    
    print(f"\n{'='*80}")
    print("LinearRegression Grid Search with temporal validation")
    print(f"{'='*80}")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )

    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol="label",
        maxIter=100
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    param_grid = (ParamGridBuilder()
                  .addGrid(lr.regParam, [0.01, 0.1])
                  .addGrid(lr.elasticNetParam, [0.5, 1.0])
                  .build())

    print(f"Parameter grid size: {len(param_grid)} combinations")

    rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    results = []
    best_model = None
    best_val_rmse = float("inf")
    best_params = None
    best_idx = None

    for i, params in enumerate(param_grid):
        current_model = pipeline.fit(train_df, params)

        val_predictions = current_model.transform(val_df)
        val_rmse = rmse_evaluator.evaluate(val_predictions)
        val_mae = mae_evaluator.evaluate(val_predictions)
        val_r2 = r2_evaluator.evaluate(val_predictions)

        test_predictions = current_model.transform(test_df)
        test_rmse = rmse_evaluator.evaluate(test_predictions)
        test_mae = mae_evaluator.evaluate(test_predictions)
        test_r2 = r2_evaluator.evaluate(test_predictions)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = current_model
            best_params = params
            best_idx = i

        results.append({
            "model_name": "LinearRegression",
            "regParam": params[lr.regParam],
            "elasticNetParam": params[lr.elasticNetParam],
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "is_best": False
        })

    if best_idx is not None:
        results[best_idx]["is_best"] = True

    return results, best_model, best_params


def train_glr_with_validation(train_df, val_df, test_df, feature_cols):
    """
    Train Generalized Linear Regression with manual grid search using validation set.
    """
    
    print(f"\n{'='*80}")
    print("GeneralizedLinearRegression Grid Search with temporal validation")
    print(f"{'='*80}")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )

    glr = GeneralizedLinearRegression(
        featuresCol="scaledFeatures",
        labelCol="label",
        family="gaussian",
        link="identity"
    )

    pipeline = Pipeline(stages=[assembler, scaler, glr])

    param_grid = (ParamGridBuilder()
                  .addGrid(glr.regParam, [0.005, 0.01, 0.02, 0.03, 0.1])
                  .addGrid(glr.maxIter, [50, 100, 150])
                  .build())

    print(f"Parameter grid size: {len(param_grid)} combinations")

    rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    results = []
    best_model = None
    best_val_rmse = float("inf")
    best_params = None
    best_idx = None

    for i, params in enumerate(param_grid):
        current_model = pipeline.fit(train_df, params)

        val_predictions = current_model.transform(val_df)
        val_rmse = rmse_evaluator.evaluate(val_predictions)
        val_mae = mae_evaluator.evaluate(val_predictions)
        val_r2 = r2_evaluator.evaluate(val_predictions)

        test_predictions = current_model.transform(test_df)
        test_rmse = rmse_evaluator.evaluate(test_predictions)
        test_mae = mae_evaluator.evaluate(test_predictions)
        test_r2 = r2_evaluator.evaluate(test_predictions)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = current_model
            best_params = params
            best_idx = i

        results.append({
            "model_name": "GeneralizedLinearRegression",
            "regParam": params[glr.regParam],
            "maxIter": params[glr.maxIter],
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "is_best": False
        })

    if best_idx is not None:
        results[best_idx]["is_best"] = True

    return results, best_model, best_params

def train_random_forest_with_validation(train_df, val_df, test_df, feature_cols):
    """
    Train Random Forest Regressor with manual grid search using validation set.
    """
    
    print(f"\n{'='*80}")
    print("RandomForestRegressor Grid Search with temporal validation")
    print(f"{'='*80}")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, rf])

    param_grid = (ParamGridBuilder()
              .addGrid(rf.numTrees, [20])
              .addGrid(rf.maxDepth, [5, 10])
              .build())

    print(f"Parameter grid size: {len(param_grid)} combinations")

    rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    results = []
    best_model = None
    best_val_rmse = float("inf")
    best_params = None
    best_idx = None

    for i, params in enumerate(param_grid):
        current_model = pipeline.fit(train_df, params)

        val_predictions = current_model.transform(val_df)
        val_rmse = rmse_evaluator.evaluate(val_predictions)
        val_mae = mae_evaluator.evaluate(val_predictions)
        val_r2 = r2_evaluator.evaluate(val_predictions)

        test_predictions = current_model.transform(test_df)
        test_rmse = rmse_evaluator.evaluate(test_predictions)
        test_mae = mae_evaluator.evaluate(test_predictions)
        test_r2 = r2_evaluator.evaluate(test_predictions)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = current_model
            best_params = params
            best_idx = i

        results.append({
            "model_name": "RandomForestRegressor",
            "numTrees": params[rf.numTrees],
            "maxDepth": params[rf.maxDepth],
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "is_best": False
        })

    if best_idx is not None:
        results[best_idx]["is_best"] = True

    return results, best_model, best_params

def evaluate_zero_baseline(val_df, test_df):
    """
    Evaluate a trivial baseline that always predicts 0.
    """

    rmse_evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    mae_evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="mae"
    )
    r2_evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="r2"
    )

    val_baseline_df = val_df.withColumn("prediction", F.lit(0.0))
    test_baseline_df = test_df.withColumn("prediction", F.lit(0.0))

    val_rmse = rmse_evaluator.evaluate(val_baseline_df)
    val_mae = mae_evaluator.evaluate(val_baseline_df)
    val_r2 = r2_evaluator.evaluate(val_baseline_df)

    test_rmse = rmse_evaluator.evaluate(test_baseline_df)
    test_mae = mae_evaluator.evaluate(test_baseline_df)
    test_r2 = r2_evaluator.evaluate(test_baseline_df)

    results = [{
        "model_name": "ZeroBaseline",
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "is_best": True
    }]

    return results


def save_results_to_csv(all_results, output_dir):
    """Convert results to DataFrame and save to CSV."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create Pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by validation RMSE, using test RMSE as secondary ordering
    results_df = results_df.sort_values(
        by=["val_rmse", "test_rmse"],
        na_position="last",
        ascending=True
    ).reset_index(drop=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"regression_results_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (All Parameter Combinations with Validation/Test Metrics)")
    print("="*80)
    print(results_df.to_string())
    
    print("\n" + "="*80)
    print("BEST MODELS (selected by Validation RMSE):")
    print("="*80)

    best_zero = results_df[
        (results_df["model_name"] == "ZeroBaseline") &
        (results_df["is_best"] == True)
    ]
    
    best_lr = results_df[
        (results_df["model_name"] == "LinearRegression") & 
        (results_df["is_best"] == True)
    ]
    best_glr = results_df[
        (results_df["model_name"] == "GeneralizedLinearRegression") & 
        (results_df["is_best"] == True)
    ]

    best_rf = results_df[
        (results_df["model_name"] == "RandomForestRegressor") & 
        (results_df["is_best"] == True)
    ]


    if not best_zero.empty:
        row = best_zero.iloc[0]
        print(f"\n✓ Zero Baseline:")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Test MAE: {row['test_mae']:.4f}")
        print(f"  Test R²: {row['test_r2']:.4f}")
        print(f"  Validation RMSE: {row['val_rmse']:.4f}")
        print(f"  Validation MAE: {row['val_mae']:.4f}")
        print(f"  Validation R²: {row['val_r2']:.4f}")
        print("  Prediction rule: always predict 0")
    
    if not best_lr.empty:
        row = best_lr.iloc[0]
        print(f"\n✓ Best Linear Regression:")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Test MAE: {row['test_mae']:.4f}")
        print(f"  Test R²: {row['test_r2']:.4f}")
        print(f"  Validation RMSE: {row['val_rmse']:.4f}")
        print(f"  Validation MAE: {row['val_mae']:.4f}")
        print(f"  Validation R²: {row['val_r2']:.4f}")
        print(f"  Parameters: regParam={row['regParam']}, elasticNetParam={row['elasticNetParam']}")
    
    if not best_glr.empty:
        row = best_glr.iloc[0]
        print(f"\n✓ Best Generalized Linear Regression:")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Test MAE: {row['test_mae']:.4f}")
        print(f"  Test R²: {row['test_r2']:.4f}")
        print(f"  Validation RMSE: {row['val_rmse']:.4f}")
        print(f"  Validation MAE: {row['val_mae']:.4f}")
        print(f"  Validation R²: {row['val_r2']:.4f}")
        print(f"  Parameters: regParam={row['regParam']}, maxIter={row['maxIter']}")

    if not best_rf.empty:
        row = best_rf.iloc[0]
        print(f"\n✓ Best Random Forest Regressor:")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Test MAE: {row['test_mae']:.4f}")
        print(f"  Test R²: {row['test_r2']:.4f}")
        print(f"  Validation RMSE: {row['val_rmse']:.4f}")
        print(f"  Validation MAE: {row['val_mae']:.4f}")
        print(f"  Validation R²: {row['val_r2']:.4f}")
        print(f"  Parameters: numTrees={row['numTrees']}, maxDepth={row['maxDepth']}")


def main():
    """Main execution function for temporal train/validation/test modeling with manual hyperparameter search."""
    
    args = get_arguments()
    spark = None

    try:
        detected_master = detect_master_argument()
    except ValueError as e:
        print(f"\n❌ CONFIGURATION ERROR: {e}")
        return
    
    print("="*80)
    print("PySpark MLlib - Stock Return Regression with Temporal Validation")
    print("Temporal train/validation/test split with manual grid search")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Master (auto-detected): {detected_master}")
    print(f"  - App Name: {args.app_name}")
    print(f"  - Num Partitions: {args.num_partitions}")
    print(f"  - Output Directory: {args.output_dir}\n")
    
    try:
        # STEP 0: Create Spark session
        print("STEP 0: Creating Spark session")
        spark = create_spark_session(args.app_name, detected_master)

        # STEP 1: Load data
        print("\nSTEP 1: Loading stock data...")
        stock_data = load_stock_data("data/Stocks", args.num_partitions)

        # STEP 1.5: Clean invalid market rows
        print("STEP 1.5: Cleaning invalid market rows...")
        stock_data = clean_market_data(stock_data)

        # STEP 1.6: Filter short symbol histories
        print("STEP 1.6: Filtering symbols with too few rows...")
        stock_data = filter_symbols_by_min_rows(stock_data, min_rows=60)
                
        # STEP 2: Add cyclic features
        print("STEP 2: Adding cyclic date features...")
        stock_data = add_cyclic_features(stock_data)
        
        # STEP 3: Add target variable
        print("STEP 3: Adding target variable (LogReturn20)...")
        stock_data = add_target_variable(stock_data)
        
        # STEP 4: Add market features
        print("STEP 4: Adding market features (returns, volatility, trend, volume)...")
        stock_data = add_market_features(stock_data)
        
        # STEP 5: Prepare features and clean data
        print("STEP 5: Preparing features (cleaning nulls)...")
        df_prepared, feature_cols = prepare_features_and_target(stock_data)
        
        # STEP 6: Create temporal train/validation/test split
        print("\nSTEP 6: Creating temporal train/validation/test split...")
        train_df, val_df, test_df = create_train_validation_test_split_by_date(
            df_prepared,
            train_end="2015-01-01",
            val_end="2016-01-01"
        )

        train_df.cache()
        val_df.cache()
        test_df.cache()

        print(f"  Train size: {train_df.count()}")
        print(f"  Validation size: {val_df.count()}")
        print(f"  Test size: {test_df.count()}")

        # STEP 6.5: Evaluate zero baseline
        print("\nSTEP 6.5: Evaluating zero-return baseline...")
        zero_baseline_results = evaluate_zero_baseline(val_df, test_df)
        
        # STEP 7: Linear Regression with temporal validation
        print("\nSTEP 7: Running Linear Regression grid search with temporal validation...")
        lr_results, lr_best_model, lr_best_params = train_linear_regression_with_validation(
            train_df, val_df, test_df, feature_cols
        )
        
        # STEP 8: Generalized Linear Regression with temporal validation
        print("\nSTEP 8: Running Generalized Linear Regression grid search with temporal validation...")
        glr_results, glr_best_model, glr_best_params = train_glr_with_validation(
            train_df, val_df, test_df, feature_cols
        )

        # # STEP 9: Random Forest with temporal validation
        # print("\nSTEP 9: Running Random Forest grid search with temporal validation...")
        # rf_results, rf_best_model, rf_best_params = train_random_forest_with_validation(
        #     train_df, val_df, test_df, feature_cols
        # )
        
        # STEP 10: Save results
        print("\nSTEP 10: Saving results...")
        all_results = zero_baseline_results + lr_results + glr_results #+ rf_results
        save_results_to_csv(all_results, args.output_dir)
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n✓ Optimizations applied:")
        print("  • ParamGridBuilder for unified parameter declaration")
        print("  • Pipelines to prevent data leakage in feature scaling")
        print("  • Temporal train/validation/test split for more realistic evaluation")
        print("  • Manual grid search on validation set")
        print("  • RegressionEvaluator for baseline, validation and final test evaluation")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
