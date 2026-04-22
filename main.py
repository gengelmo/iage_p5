"""
PySpark MLlib Script for Stock Price Prediction with Grid Search
Trains two different regression models (Linear Regression and MLP) on stock price data
with grid search over hyperparameters.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import math
import re
import socket

import pandas as pd
from pyspark.sql import SparkSession, Window, functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np


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
        description="Train regression models on stock price data using PySpark MLlib"
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
        default="StockPriceRegression",
        help="Spark application name (default: StockPriceRegression)",
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


#def load_stock_data(data_dir: str, num_partitions: int):
#    """Load all stock CSV files from data_dir into a single Spark DataFrame."""
#    spark = SparkSession.getActiveSession()
#    
#    data_path = Path(data_dir)
#    txt_files = sorted(data_path.glob("*.txt"))
#    
#    if not txt_files:
#        raise ValueError(f"No .txt files found in {data_dir}")
#    
#    print(f"Found {len(txt_files)} stock files. Loading...")
#    
#    # Read first file to infer schema
#    first_file = str(txt_files[0])
#    df = spark.read.option("header", "true").option("inferSchema", "true").csv(first_file)
#    
#    # Add Symbol column
#    df = df.withColumn("Symbol", F.lit(txt_files[0].name.replace(".us.txt", "")))
#    
#    # Read and process remaining files
#    for file_path in txt_files[1:]:
#        temp_df = spark.read.option("header", "true").option("inferSchema", "true").csv(str(file_path))
#        temp_df = temp_df.withColumn("Symbol", F.lit(file_path.name.replace(".us.txt", "")))
#        df = df.union(temp_df)
#    
#    # Repartition for parallel processing
#    df = df.repartition(num_partitions)
#    
#    # Convert Date to timestamp and numeric columns
#    df = df.withColumn("Date", F.to_timestamp("Date"))
#    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "OpenInt"]
#    for col in numeric_cols:
#        df = df.withColumn(col, F.col(col).cast(T.DoubleType()))
#    
#    print(f"Loaded data with {df.count()} rows")
#    return df

# def load_stock_data(data_dir: str, num_partitions: int):
#     """Carga archivos CSV saltando aquellos que no tengan el número correcto de columnas."""
#     spark = SparkSession.getActiveSession()
#     data_path = Path(data_dir)
#     txt_files = sorted(data_path.glob("*.txt"))
#     
#     if not txt_files:
#         raise ValueError(f"No se encontraron archivos .txt en {data_dir}")
#     
#     print(f"Buscando en {len(txt_files)} archivos... Validando esquemas.")
#     
#     # 1. Definimos las columnas que esperamos basándonos en el estándar (7 columnas originales)
#     # Date, Open, High, Low, Close, Volume, OpenInt
#     EXPECTED_CSV_COL_COUNT = 7 
# 
#     df = None
#     files_loaded = 0
#     files_skipped = 0
# 
#     for file_path in txt_files:
#         # Leemos el archivo temporalmente
#         temp_df = spark.read.option("header", "true").option("inferSchema", "true").csv(str(file_path))
#         
#         # 2. Verificamos si el archivo tiene las columnas correctas
#         if len(temp_df.columns) == EXPECTED_CSV_COL_COUNT:
#             print(file_path, end=", ")
#             temp_df = temp_df.withColumn("Symbol", F.lit(file_path.name.replace(".us.txt", "")))
#             
#             if df is None:
#                 df = temp_df
#             else:
#                 df = df.union(temp_df)
#             files_loaded += 1
#         else:
#             print(file_path, " VACIO", end=", ")
#             # Si no coincide, lo ignoramos y avisamos
#             files_skipped += 1
#             # Opcional: imprimir el nombre del archivo problemático
#             # print(f"Skipping {file_path.name}: found {len(temp_df.columns)} columns.")
# 
#     if df is None:
#         raise ValueError("No se pudo cargar ningún archivo válido.")
# 
#     print(f"Carga finalizada: {files_loaded} archivos cargados, {files_skipped} archivos saltados.")
#     
#     # Reparticionar y procesar tipos
#     df = df.repartition(num_partitions)
#     df = df.withColumn("Date", F.to_timestamp("Date"))
#     numeric_cols = ["Open", "High", "Low", "Close", "Volume", "OpenInt"]
#     for col in numeric_cols:
#         df = df.withColumn(col, F.col(col).cast(T.DoubleType()))
#     
#     print(f"Dataset total creado con {df.count()} filas.")
#     return df

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


def add_cyclic_features(df):
    """Add cyclic features for date components (month and day only)."""
    
    # Cyclic encoding for month (0-11 range for trigonometry)
    month_normalized = (F.month("Date") - 1) / 12.0 * 2 * math.pi
    df = df.withColumn("DateMonthSin", F.sin(month_normalized))
    df = df.withColumn("DateMonthCos", F.cos(month_normalized))
    
    # Cyclic encoding for day (0-30 range for trigonometry)
    day_normalized = (F.dayofmonth("Date") - 1) / 31.0 * 2 * math.pi
    df = df.withColumn("DateDaySin", F.sin(day_normalized))
    df = df.withColumn("DateDayCos", F.cos(day_normalized))
    
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


def add_past_returns(df):
    """Add past return features at 3, 5, and 10 days normalized by current close."""
    
    window_spec = Window.partitionBy("Symbol").orderBy("Date")
    
    # Calculate log-returns relative to current close price at lags 3, 5, 10 days
    df = df.withColumn(
        "PastReturn3",
        F.log(F.lag("Close", 3).over(window_spec) / F.col("Close"))
    )
    df = df.withColumn(
        "PastReturn5",
        F.log(F.lag("Close", 5).over(window_spec) / F.col("Close"))
    )
    df = df.withColumn(
        "PastReturn10",
        F.log(F.lag("Close", 10).over(window_spec) / F.col("Close"))
    )
    
    return df


def prepare_features_and_target(df):
    """Select and clean data for modeling (remove nulls)."""
    
    feature_cols = [
        "DateMonthSin", "DateMonthCos", "DateDaySin", "DateDayCos",
        "PastReturn3", "PastReturn5", "PastReturn10"
    ]
    
    # Remove rows with null target (last 20 days of each stock)
    df = df.dropna(subset=["LogReturn20"])
    
    # Remove rows with any null features
    df = df.dropna(subset=feature_cols)
    
    # Select final columns (target renamed to label for MLlib)
    df = df.select(feature_cols + ["LogReturn20"])
    df = df.withColumnRenamed("LogReturn20", "label")
    
    return df, feature_cols


def create_train_test_split(df, test_ratio=0.2, seed=42):
    """Create train/test split using native Spark randomSplit (distributed approach)."""
    train_ratio = 1.0 - test_ratio
    train_df, test_df = df.randomSplit([train_ratio, test_ratio], seed=seed)
    
    return train_df, test_df


def train_linear_regression_with_crossvalidation(train_df, test_df, feature_cols, parallelism=4):
    """
    Train Linear Regression with grid search using CrossValidator for parallel hyperparameter tuning.
    
    Args:
        train_df: Training data with feature columns and 'label' target
        test_df: Test data for final evaluation
        feature_cols: List of feature column names
        parallelism: Number of parallel hyperparameter configurations to train
    
    Returns:
        List of results dictionaries with metrics and hyperparameters
    """
    
    print(f"\n{'='*80}")
    print("LinearRegression Grid Search with CrossValidator (parallelism={})".format(parallelism))
    print(f"{'='*80}")
    
    # Stage 1: VectorAssembler (combines feature columns into 'features' vector)
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    # Stage 2: StandardScaler (scales features - will be re-fit on each fold)
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )
    
    # Stage 3: Model
    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol="label",
        maxIter=100,
        # standardization=False  # Already scaled by StandardScaler
    )
    
    # Build pipeline: Assembler -> Scaler -> Model
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    
    # Define parameter grid for grid search
    param_grid = (ParamGridBuilder()
                  .addGrid(lr.regParam, [0.01, 0.1])
                  .addGrid(lr.elasticNetParam, [0.5, 1.0])
                  .build())
    
    print(f"Parameter grid size: {len(param_grid)} combinations")
    
    # Define evaluation metric
    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # CrossValidator: automatic k-fold cross-validation with parallel hyperparameter tuning
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,  # 3-fold cross-validation
        parallelism=parallelism,  # Train up to N hyperparameter configs simultaneously
        seed=42
    )
    
    print("Starting CrossValidator (may take a few minutes)...\n")
    cv_model = cv.fit(train_df)
    
    # Extract best model
    best_pipeline_model = cv_model.bestModel
    best_params = cv_model.bestModel.stages[-1].extractParamMap()
    
    # Evaluate on test set
    test_predictions = best_pipeline_model.transform(test_df)
    
    rmse = evaluator.evaluate(test_predictions, {evaluator.metricName: "rmse"})
    mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(test_predictions)
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = r2_evaluator.evaluate(test_predictions)
    
    # Collect CV scores for all parameter combinations
    results = []
    for i, (params, avg_cv_score) in enumerate(zip(param_grid, cv_model.avgMetrics)):
        result = {
            "model_name": "LinearRegression",
            "regParam": params[lr.regParam],
            "elasticNetParam": params[lr.elasticNetParam],
            "cv_rmse_avg": avg_cv_score,
            "test_rmse": rmse if params == best_params else None,
            "test_mae": mae if params == best_params else None,
            "test_r2": r2 if params == best_params else None,
            "is_best": (params == best_params)
        }
        results.append(result)
    
    return results, best_pipeline_model, cv_model


def train_glr_with_crossvalidation(train_df, test_df, feature_cols, parallelism=4):
    """
    Train Generalized Linear Regression with grid search using CrossValidator for parallel hyperparameter tuning.
    
    Args:
        train_df: Training data with feature columns and 'label' target
        test_df: Test data for final evaluation
        feature_cols: List of feature column names
        parallelism: Number of parallel hyperparameter configurations to train
    
    Returns:
        List of results dictionaries with metrics and hyperparameters
    """
    
    print(f"\n{'='*80}")
    print("GeneralizedLinearRegression Grid Search with CrossValidator (parallelism={})".format(parallelism))
    print(f"{'='*80}")
    
    # Stage 1: VectorAssembler
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    # Stage 2: StandardScaler
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )
    
    # Stage 3: Model
    glr = GeneralizedLinearRegression(
        featuresCol="scaledFeatures",
        labelCol="label",
        family="gaussian",
        link="identity",
        # solver="normal",
        # standardization=False  # Already scaled by StandardScaler
    )
    
    # Build pipeline
    pipeline = Pipeline(stages=[assembler, scaler, glr])
    
    # Define parameter grid
    param_grid = (ParamGridBuilder()
                  .addGrid(glr.regParam, [0.01, 0.1])
                  .addGrid(glr.maxIter, [50, 100])
                  .build())
    
    print(f"Parameter grid size: {len(param_grid)} combinations")
    
    # Define evaluation metric
    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # CrossValidator
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=parallelism,
        seed=42
    )
    
    print("Starting CrossValidator (may take a few minutes)...\n")
    cv_model = cv.fit(train_df)
    
    # Extract best model
    best_pipeline_model = cv_model.bestModel
    best_params = cv_model.bestModel.stages[-1].extractParamMap()
    
    # Evaluate on test set
    test_predictions = best_pipeline_model.transform(test_df)
    
    rmse = evaluator.evaluate(test_predictions, {evaluator.metricName: "rmse"})
    mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(test_predictions)
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = r2_evaluator.evaluate(test_predictions)
    
    # Collect CV scores for all parameter combinations
    results = []
    for i, (params, avg_cv_score) in enumerate(zip(param_grid, cv_model.avgMetrics)):
        result = {
            "model_name": "GeneralizedLinearRegression",
            "regParam": params[glr.regParam],
            "maxIter": params[glr.maxIter],
            "cv_rmse_avg": avg_cv_score,
            "test_rmse": rmse if params == best_params else None,
            "test_mae": mae if params == best_params else None,
            "test_r2": r2 if params == best_params else None,
            "is_best": (params == best_params)
        }
        results.append(result)
    
    return results, best_pipeline_model, cv_model


def save_results_to_csv(all_results, output_dir):
    """Convert results to DataFrame and save to CSV."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create Pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by test_rmse (best results first), using CV RMSE as tiebreaker
    results_df = results_df.sort_values(
        by="test_rmse", 
        na_position='last',
        ascending=True
    ).reset_index(drop=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"regression_results_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (All Parameter Combinations with CV Metrics)")
    print("="*80)
    print(results_df.to_string())
    
    print("\n" + "="*80)
    print("BEST MODELS (by Test RMSE):")
    print("="*80)
    
    best_lr = results_df[
        (results_df["model_name"] == "LinearRegression") & 
        (results_df["is_best"] == True)
    ]
    best_glr = results_df[
        (results_df["model_name"] == "GeneralizedLinearRegression") & 
        (results_df["is_best"] == True)
    ]
    
    if not best_lr.empty:
        row = best_lr.iloc[0]
        print(f"\n✓ Best Linear Regression:")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Test MAE: {row['test_mae']:.4f}")
        print(f"  Test R²: {row['test_r2']:.4f}")
        print(f"  CV RMSE Average: {row['cv_rmse_avg']:.4f}")
        print(f"  Parameters: regParam={row['regParam']}, elasticNetParam={row['elasticNetParam']}")
    
    if not best_glr.empty:
        row = best_glr.iloc[0]
        print(f"\n✓ Best Generalized Linear Regression:")
        print(f"  Test RMSE: {row['test_rmse']:.4f}")
        print(f"  Test MAE: {row['test_mae']:.4f}")
        print(f"  Test R²: {row['test_r2']:.4f}")
        print(f"  CV RMSE Average: {row['cv_rmse_avg']:.4f}")
        print(f"  Parameters: regParam={row['regParam']}, maxIter={row['maxIter']}")


def main():
    """Main execution function with optimized parallel grid search."""
    
    args = get_arguments()
    spark = None

    try:
        detected_master = detect_master_argument()
    except ValueError as e:
        print(f"\n❌ CONFIGURATION ERROR: {e}")
        return
    
    print("="*80)
    print("PySpark MLlib - Stock Price Regression with CrossValidator")
    print("Optimized for parallel hyperparameter tuning")
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
        
        # STEP 2: Add cyclic features
        print("STEP 2: Adding cyclic date features...")
        stock_data = add_cyclic_features(stock_data)
        
        # STEP 3: Add target variable
        print("STEP 3: Adding target variable (LogReturn20)...")
        stock_data = add_target_variable(stock_data)
        
        # STEP 4: Add past return features
        print("STEP 4: Adding past return features (3, 5, 10 days)...")
        stock_data = add_past_returns(stock_data)
        
        # STEP 5: Prepare features and clean data
        print("STEP 5: Preparing features (cleaning nulls)...")
        df_prepared, feature_cols = prepare_features_and_target(stock_data)
        
        # STEP 6: Create train/test split using native randomSplit
        print("\nSTEP 6: Creating train/test split (80/20) using randomSplit...")
        train_df, test_df = create_train_test_split(df_prepared, test_ratio=0.2, seed=42)
        train_df.cache()
        test_df.cache()
        print(f"  Train size: {train_df.count()}, Test size: {test_df.count()}")
        
        # STEP 7: Linear Regression with CrossValidator (parallelism=4)
        print("\nSTEP 7: Running Linear Regression grid search with CrossValidator...")
        lr_results, lr_best_model, lr_cv = train_linear_regression_with_crossvalidation(
            train_df, test_df, feature_cols, parallelism=4
        )
        
        # STEP 8: Generalized Linear Regression with CrossValidator (parallelism=4)
        print("\nSTEP 8: Running Generalized Linear Regression grid search with CrossValidator...")
        glr_results, glr_best_model, glr_cv = train_glr_with_crossvalidation(
            train_df, test_df, feature_cols, parallelism=4
        )
        
        # STEP 9: Save results
        print("\nSTEP 9: Saving results...")
        all_results = lr_results + glr_results
        save_results_to_csv(all_results, args.output_dir)
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n✓ Optimizations applied:")
        print("  • ParamGridBuilder for unified parameter declaration")
        print("  • CrossValidator for parallel hyperparameter training")
        print("  • Pipelines to prevent data leakage (StandardScaler re-fitted per fold)")
        print("  • randomSplit() for distributed train/test split (no global Window operations)")
        print("  • Integrated RegressionEvaluator for automated cross-fold validation")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
